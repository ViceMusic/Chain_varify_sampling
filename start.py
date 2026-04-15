import os
import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from utils import *


# =========================================================
# 手动配置区：直接改这里
# =========================================================
CONFIG = {
    # 基础
    "method": "DC",
    "dataset": "MODELNET40_H5",      # 这里填你新增的 h5 数据集名
    "model": "PointNet",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 蒸馏设置
    "ppc": 10,
    "eval_mode": "S",
    "num_exp": 1,
    "num_eval": 10,                   # 测试时建议先小一点
    "epoch_eval_train": 500,         # 测试时建议先小一点
    "Iteration": 1500,                 # 测试先小一点，正式跑再调大
    "lr_img": 10.0,
    "lr_rot": 0.01,
    "lr_net": 0.01,
    "batch_real": 8,
    "batch_train": 8,
    "init": "real",
    "feature_transform": 0,
    "mmdkernel": "gaussian",

    # 保存
    "save_path": "result_h5",
    "mode": "result_h5",
    "addition_setting": "None",

    # 调试 / 测试模式
    "TEST_ONLY": False,               # True: 只做快速检查；False: 正式训练
    "TEST_EVAL_IT": 0,               # 测试模式下在第几轮做评估
    "SAVE_SYNTHETIC_TXT": False,     # 是否保存中间 txt
    "LOG_CLASS_INFO": False,         # 是否打印每类样本数

    # 训练过程输出频率
    "PRINT_EVERY": 10,               # 每多少轮 print 一次训练 loss
}


def build_args_from_config(config_dict):
    return SimpleNamespace(**config_dict)


def set_seed(seed=1999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_rotation_matrix(angle, rot_type="x"):
    c = torch.cos(angle)
    s = torch.sin(angle)
    zeros = torch.zeros_like(c)
    ones = torch.ones_like(c)

    if rot_type == "x":
        return torch.stack([
            torch.stack([ones, zeros, zeros], dim=1),
            torch.stack([zeros, c, -s], dim=1),
            torch.stack([zeros, s, c], dim=1)
        ], dim=2)
    elif rot_type == "y":
        return torch.stack([
            torch.stack([c, zeros, s], dim=1),
            torch.stack([zeros, ones, zeros], dim=1),
            torch.stack([-s, zeros, c], dim=1)
        ], dim=2)
    else:
        return torch.stack([
            torch.stack([c, -s, zeros], dim=1),
            torch.stack([s, c, zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=2)


def save_pointcloud_txt_batch(pointcloud_batch, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    for i, save_pc in enumerate(pointcloud_batch):
        file_name = os.path.join(save_dir, f"{prefix}_{i}.txt")
        np.savetxt(file_name, save_pc.T, delimiter=",")


def resolve_dataset(args):
    """
    优先尝试：
    1. get_dataset_dispatch
    2. get_dataset_h5 （当 dataset 以 _H5 结尾）
    3. 回退到 get_dataset
    """
    if "get_dataset_dispatch" in globals():
        return get_dataset_dispatch(args, args.dataset)
    elif "get_dataset_h5" in globals() and str(args.dataset).endswith("_H5"):
        return get_dataset_h5(args, args.dataset)
    else:
        return get_dataset(args, args.dataset)


def print_train_progress(it, total_it, loss_avg):
    percent = 100.0 * it / max(total_it, 1)
    print(f"[{percent:6.2f}%] Iter {it:04d}/{total_it} | loss = {loss_avg:.4f}")


def main():
    args = build_args_from_config(CONFIG)

    os.makedirs(args.mode, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    log_filename = (
        f"{args.mode}/"
        f"LRimg{args.lr_img}_LRnet{args.lr_net}_ppc{args.ppc}_"
        f"Model_{args.model}_It{args.Iteration}_Dataset_{args.dataset}_init_{args.init}"
    )
    if args.addition_setting != "None":
        log_filename += f"_{args.addition_setting}"

    logger = build_logger(".", log_filename)

    logger.info("Device: %s", args.device)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Model: %s", args.model)
    logger.info("TEST_ONLY: %s", args.TEST_ONLY)

    # =========================================================
    # 数据集读取
    # =========================================================
    npoints, coord_dim, num_classes, dst_train, _, testloader = resolve_dataset(args)

    args.num_classes = num_classes
    model_eval_pool = get_eval_pool(args.eval_mode, args.model)

    # 测试模式：只在指定轮次评估一次
    if args.TEST_ONLY:
        eval_it_pool = [args.TEST_EVAL_IT]
    else:
        eval_it_pool = (
            np.arange(0, args.Iteration + 1, 250).tolist()
            if args.eval_mode in ["S", "SSS"]
            else [args.Iteration]
        )

    logger.info("eval_it_pool: %s", eval_it_pool)

    accs_all_exps = {key: [] for key in model_eval_pool}
    accs_all = []
    data_save = []

    for exp in range(args.num_exp):
        logger.info("\n================== Exp %d ==================\n", exp)

        # =========================================================
        # 整理真实数据
        # =========================================================
        pointcloud_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))]
        indices_class = [[] for _ in range(num_classes)]

        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        pointcloud_all = torch.cat(pointcloud_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        if args.LOG_CLASS_INFO:
            for c in range(num_classes):
                logger.info("class %d: %d samples", c, len(indices_class[c]))

        def get_pointclouds(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return pointcloud_all[idx_shuffle]

        channel = 3

        # =========================================================
        # 初始化 synthetic 点云
        # =========================================================
        pointcloud_tmp = torch.randn(
            size=(num_classes * args.ppc, coord_dim, npoints),
            dtype=torch.float,
            requires_grad=True,
            device=args.device
        )
        pointcloud_tmp = torch.tanh(pointcloud_tmp).detach().clone().requires_grad_(True)

        label_syn = torch.tensor(
            [np.ones(args.ppc) * i for i in range(num_classes)],
            dtype=torch.int,
            requires_grad=False,
            device=args.device
        ).view(-1)

        theta_x = torch.zeros(num_classes * args.ppc, dtype=torch.float, device=args.device).requires_grad_(True)
        theta_y = torch.zeros(num_classes * args.ppc, dtype=torch.float, device=args.device).requires_grad_(True)
        theta_z = torch.zeros(num_classes * args.ppc, dtype=torch.float, device=args.device).requires_grad_(True)

        optimizer_theta = torch.optim.SGD([
            {"params": theta_x, "lr": args.lr_rot / 10},
            {"params": theta_y, "lr": args.lr_rot},
            {"params": theta_z, "lr": args.lr_rot / 10},
        ], momentum=0.5)

        if args.init == "real":
            logger.info("Initialize synthetic data from random real samples")
            for c in range(num_classes):
                pointcloud_tmp.data[c * args.ppc:(c + 1) * args.ppc] = get_pointclouds(c, args.ppc).detach().data
        else:
            logger.info("Initialize synthetic data from random noise")

        # =========================================================
        # 可选：保存初始 synthetic 数据
        # =========================================================
        pc_div_name = (
            f"{args.mode}/LRimg{args.lr_img}_LRnet{args.lr_net}_ppc{args.ppc}_"
            f"Model_{args.model}_It{args.Iteration}_Dataset_{args.dataset}_init_{args.init}"
        )
        if args.addition_setting != "None":
            pc_div_name += f"_{args.addition_setting}"

        if args.SAVE_SYNTHETIC_TXT:
            os.makedirs(pc_div_name, exist_ok=True)
            for c in range(num_classes):
                label_folder = os.path.join(pc_div_name, f"class_{c}")
                os.makedirs(label_folder, exist_ok=True)
                pc_per_class = pointcloud_tmp.data[c * args.ppc:(c + 1) * args.ppc].cpu().numpy()
                save_pointcloud_txt_batch(pc_per_class, label_folder, f"init_{c}")

        # =========================================================
        # 训练准备
        # =========================================================
        optimizer_img = torch.optim.SGD([pointcloud_tmp], lr=args.lr_img, momentum=0.5)
        criterion = nn.CrossEntropyLoss().to(args.device)
        m3d_criterion = M3DLoss(args.mmdkernel)

        logger.info("%s training begins", get_time())

        # =========================================================
        # 主训练循环
        # =========================================================
        for it in range(args.Iteration + 1):
            with torch.no_grad():
                theta_x.data = torch.remainder(theta_x.data + math.pi, 2 * math.pi) - math.pi
                theta_y.data = torch.remainder(theta_y.data + math.pi, 2 * math.pi) - math.pi
                theta_z.data = torch.remainder(theta_z.data + math.pi, 2 * math.pi) - math.pi

            rot_matrix_x = create_rotation_matrix(theta_x, "x")
            rot_matrix_y = create_rotation_matrix(theta_y, "y")
            rot_matrix_z = create_rotation_matrix(theta_z, "z")
            trans_matrix = torch.bmm(rot_matrix_z, torch.bmm(rot_matrix_y, rot_matrix_x))

            pc_transposed = pointcloud_tmp.permute(0, 2, 1).contiguous()
            pc_rotated = torch.bmm(pc_transposed, trans_matrix)
            pointcloud_syn = pc_rotated.permute(0, 2, 1).contiguous()

            # =====================================================
            # 评估 synthetic 数据
            # =====================================================
            if it in eval_it_pool:
                logger.info("Evaluate at iteration %d", it)

                for model_eval in model_eval_pool:
                    accs = []
                    accs_per_class = []

                    for it_eval in range(args.num_eval):
                        print(f"[Eval] model={model_eval} | round {it_eval + 1}/{args.num_eval}")
                        torch.manual_seed(1996 + it_eval)

                        net_eval = get_network(model_eval, channel, num_classes, args.feature_transform).to(args.device)
                        pointcloud_syn_eval = copy.deepcopy(pointcloud_syn.detach())
                        label_syn_eval = copy.deepcopy(label_syn.detach())

                        _, _, acc_test, acc_test_per_class, predictions_per_sample = evaluate_synset(
                            it_eval,
                            net_eval,
                            pointcloud_syn_eval,
                            label_syn_eval,
                            testloader,
                            args
                        )
                        accs.append(acc_test)

                        if len(accs_per_class) == 0:
                            accs_per_class = [[] for _ in range(num_classes)]

                        for class_idx in range(num_classes):
                            accs_per_class[class_idx].append(acc_test_per_class[class_idx])

                    mean_acc = float(np.mean(accs))
                    std_acc = float(np.std(accs))
                    logger.info("Model %s | mean acc = %.4f | std = %.4f", model_eval, mean_acc, std_acc)

                    accs_all.append(mean_acc)
                    if it == args.Iteration or args.TEST_ONLY:
                        accs_all_exps[model_eval] += accs

                if args.SAVE_SYNTHETIC_TXT:
                    pointcloud_syn_vis = copy.deepcopy(pointcloud_syn.detach().cpu().numpy())
                    for c in range(num_classes):
                        label_folder = os.path.join(pc_div_name, f"class_{c}")
                        os.makedirs(label_folder, exist_ok=True)
                        pc_syn_per_class = pointcloud_syn_vis[c * args.ppc:(c + 1) * args.ppc]
                        save_pointcloud_txt_batch(pc_syn_per_class, label_folder, f"iter_{it}_class_{c}")

                if args.TEST_ONLY:
                    logger.info("TEST_ONLY=True, stop after first evaluation.")
                    break

            # =====================================================
            # 更新 synthetic 数据
            # =====================================================
            net = get_network(args.model, channel, num_classes, args.feature_transform).to(args.device)
            net.train()

            for param in net.parameters():
                param.requires_grad = False

            bn_flag = any("BatchNorm" in module._get_name() for module in net.modules())
            if bn_flag:
                bn_size_pc = 8
                pc_real = torch.cat([get_pointclouds(c, bn_size_pc) for c in range(num_classes)], dim=0)
                net.train()
                _ = net(pc_real)
                for module in net.modules():
                    if "BatchNorm" in module._get_name():
                        module.eval()

            loss = torch.tensor(0.0, device=args.device)

            for c in range(num_classes):
                pc_syn = pointcloud_syn[c * args.ppc:(c + 1) * args.ppc].reshape(
                    args.ppc, coord_dim, npoints
                ).to(args.device)
                pc_real = get_pointclouds(c, args.batch_real)

                with torch.no_grad():
                    _, _, _, _, layers_real = net(pc_real)

                _, _, _, _, layers_syn = net(pc_syn)

                sorted_real = torch.sort(layers_real["x_m"], dim=2, descending=True)[0].detach()
                sorted_syn = torch.sort(layers_syn["x_m"], dim=2, descending=True)[0]

                real = sorted_real.mean(dim=0)
                syn = sorted_syn.mean(dim=0)

                loss1 = (((real - syn) ** 2).sum(dim=0)).mean() * 0.2
                loss1 += m3d_criterion(layers_real["x_gf"], layers_syn["x_gf"]) * 0.001
                loss += loss1 * args.ppc

            optimizer_img.zero_grad()
            optimizer_theta.zero_grad()
            loss.backward()
            optimizer_img.step()
            optimizer_theta.step()

            loss_avg = loss.item() / num_classes

            if (it % args.PRINT_EVERY == 0) or (it == args.Iteration):
                print_train_progress(it, args.Iteration, loss_avg)

            if it == args.Iteration:
                data_save.append([
                    copy.deepcopy(pointcloud_syn.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu())
                ])
                torch.save(
                    {"data": data_save, "accs_all_exps": accs_all_exps},
                    os.path.join(args.save_path, f"res_{args.method}_{args.dataset}_{args.model}_{args.ppc}ppc.pt")
                )

    logger.info("\n==================== Final Results ====================\n")
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        if len(accs) > 0:
            logger.info(
                "Run %d exp | train on %s | eval %d random %s | mean=%.2f%% std=%.2f%%",
                args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100
            )

    if len(accs_all) > 0:
        print("\n==================== Final Results ====================\n")
        print(np.array(accs_all))


if __name__ == "__main__":
    set_seed(1999)
    main()