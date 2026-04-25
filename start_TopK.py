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
    "dataset": "MODELNET40_H5",
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
    "SPATIAL_ALIGN_MODE": "soft",     # "soft" 或 "hard"

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
    "message":"test" #这个变量用来给生成结果做标识的
}

# 逐个方法是协助读取CONFIG的，不用管
def build_args_from_config(config_dict):
    return SimpleNamespace(**config_dict)

# 设置随机函数
def set_seed(seed=1999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 这东西是在构造旋转矩阵
# 也就是给定一个轴以后，生成绕这个轴旋转的矩阵，输入是旋转角度和旋转轴的类型（x/y/z）
# 虽然是在计算图之中，但是本身其实并不存在可以训练的参数，就是一个纯粹地计算函数
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

# 保存为txt
def save_pointcloud_txt_batch(pointcloud_batch, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    for i, save_pc in enumerate(pointcloud_batch):
        file_name = os.path.join(save_dir, f"{prefix}_{i}.txt")
        np.savetxt(file_name, save_pc.T, delimiter=",")

# 打印进度
def print_train_progress(it, total_it, loss_avg):
    percent = 100.0 * it / max(total_it, 1)
    print(f"[{percent:6.2f}%] Iter {it:04d}/{total_it} | loss = {loss_avg:.4f}")


def main():
    args = build_args_from_config(CONFIG)

    # 获取配置和打印一些东西
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
    npoints, coord_dim, num_classes, dst_train, _, testloader = get_dataset(args, args.dataset)

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
        # 生成数据格式为[num_classes * ppc, 3, npoints]
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
        # iteration外循环内容是每轮训练都要做的事情，主要是更新 synthetic 数据和评估 synthetic 数据
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

            # 初始化信息
            loss = torch.tensor(0.0, device=args.device)
            # 按照每个类别进行循环
            for c in range(num_classes):
                # 取出所有的合成数据
                pc_syn = pointcloud_syn[c * args.ppc:(c + 1) * args.ppc].reshape(
                    args.ppc, coord_dim, npoints
                ).to(args.device)
                # 随机取batch_real个真实数据
                pc_real = get_pointclouds(c, args.batch_real)

                #提取特征，这个net不是模型？？
                with torch.no_grad():
                    _, _, _, _, layers_real = net(pc_real)
                _, _, _, _, layers_syn = net(pc_syn)

                # =====================================================
                # 原始 SADM 部分：channel-wise sorted feature matching
                # =====================================================
                # layers_real["x_m"] / layers_syn["x_m"] 的形状是 [B, C, N]
                # B: batch 内点云数量
                # C: 网络映射后的 feature channel 数
                # N: 每个点云中的点数
                #
                # 这里按照 dim=2 排序，也就是在“每个 channel 内部”对 N 个点的响应值排序。
                # 注意：排序后，不同 channel 的同一个 rank 位置不再对应同一个空间点。
                # 因此这里比较的不是点对点结构，而是每个 channel 内部的“响应分布形状”。
                sorted_real = torch.sort(layers_real["x_m"], dim=2, descending=True)[0].detach()
                sorted_syn = torch.sort(layers_syn["x_m"], dim=2, descending=True)[0]

                # 对整个 batch 做平均，得到 real/syn 的 channel-wise response distribution
                real = sorted_real.mean(dim=0)
                syn = sorted_syn.mean(dim=0)

                # 原始 SADM feature distribution matching loss
                loss1 = (((real - syn) ** 2).sum(dim=0)).mean() * 0.2
                # 空间增强作用的损失，初始值为 None，后面根据选择的模式计算得到
                loss_spatial = None


                # =====================================================
                # 新增：response-guided spatial alignment
                # =====================================================
                # 这部分不在排序后的 feature 上操作，而是在排序之前的 x_m 上计算每个点的重要性。
                # layers_*["x_m"]: [B, C, N]
                # 对 C 个 channel 求平均，得到每个点的总体响应分数 response: [B, N]
                #
                # 直观理解：
                # response 高的点 = 当前网络更“兴奋”、更关注的点
                # response 低的点 = 当前网络不太关注的点
                #
                # 然后把这些高响应点重新映射回原始 xyz 空间 pc_real / pc_syn，
                # 让 real 和 syn 的“关键响应区域”在空间上也尽量接近。
                response_real = layers_real["x_m"].detach().mean(dim=1)  # [B, N]
                response_syn = layers_syn["x_m"].mean(dim=1)             # [B, N]

                if args.SPATIAL_ALIGN_MODE == "soft":
                    # -------------------------------------------------
                    # soft 模式：响应加权空间中心
                    # -------------------------------------------------
                    # softmax 后，每个点都有一个权重。
                    # 高响应点权重大，低响应点权重小。
                    # 这相当于计算“模型关注区域”的加权中心。
                    wr = torch.softmax(response_real, dim=1).unsqueeze(1)  # [B, 1, N]
                    ws = torch.softmax(response_syn, dim=1).unsqueeze(1)   # [B, 1, N]

                    # pc_real / pc_syn: [B, 3, N]
                    # 加权求和后得到每个点云的响应中心: [B, 3]
                    real_center = (pc_real * wr).sum(dim=2)  # [B, 3]
                    syn_center = (pc_syn * ws).sum(dim=2)    # [B, 3]

                    # batch 维度再求平均，比较 real/syn 的整体响应中心
                    loss_spatial = torch.mean(
                        (real_center.mean(dim=0) - syn_center.mean(dim=0)) ** 2
                    )

                elif args.SPATIAL_ALIGN_MODE == "hard":
                    # -------------------------------------------------
                    # hard 模式：Top-k 高响应点空间中心
                    # -------------------------------------------------
                    # 只选择 response 最高的前 k 个点。
                    # 这一步是真正的“选点”，idx 仍然对应原始点云中的点索引。
                    topk_ratio = 0.2
                    k = max(1, int(topk_ratio * response_real.size(1)))

                    idx_real = torch.topk(response_real, k=k, dim=1).indices  # [B, k]
                    idx_syn = torch.topk(response_syn, k=k, dim=1).indices    # [B, k]

                    # 为了从 pc_real / pc_syn: [B, 3, N] 中取点，
                    # 需要把 idx 扩展成 [B, 3, k]
                    idx_real_xyz = idx_real.unsqueeze(1).expand(-1, 3, -1)
                    idx_syn_xyz = idx_syn.unsqueeze(1).expand(-1, 3, -1)

                    # 根据 top-k 索引，回到原始 xyz 空间中取出高响应点
                    real_key_xyz = torch.gather(pc_real, dim=2, index=idx_real_xyz)  # [B, 3, k]
                    syn_key_xyz = torch.gather(pc_syn, dim=2, index=idx_syn_xyz)     # [B, 3, k]

                    # 计算这些高响应点的空间中心
                    real_center = real_key_xyz.mean(dim=2)  # [B, 3]
                    syn_center = syn_key_xyz.mean(dim=2)    # [B, 3]

                    # 比较 real/syn 高响应区域的空间中心
                    loss_spatial = torch.mean(
                        (real_center.mean(dim=0) - syn_center.mean(dim=0)) ** 2
                    )

                else:
                    raise ValueError(f"Unknown SPATIAL_ALIGN_MODE: {args.SPATIAL_ALIGN_MODE}")

                # 空间响应对齐损失权重，建议先小一点
                loss1 += loss_spatial * 0.05

                # =====================================================
                # 保留原始 M3D loss
                # =====================================================
                # x_gf 是 global feature，M3D loss 用于进一步约束 real/syn 的整体特征分布。
                loss1 += m3d_criterion(layers_real["x_gf"], layers_syn["x_gf"]) * 0.001

                # 按 ppc 放大，与原始代码保持一致
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
                    os.path.join(args.save_path, f"res_{args.method}_{args.dataset}_{args.model}_{args.ppc}ppc_{args.message}.pt")
                )
    # 下面纯打印来着
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
