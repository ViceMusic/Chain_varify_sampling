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
    "multieval": False,
    "num_exp": 1,
    "num_eval": 10,
    "epoch_eval_train": 500,
    "Iteration": 4000,
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
    "TEST_ONLY": False,
    "TEST_EVAL_IT": 0,
    "SAVE_SYNTHETIC_TXT": False,
    "LOG_CLASS_INFO": False,

    # 训练过程输出频率
    "PRINT_EVERY": 10,

    # =====================================================
    # Adaptive Multi-layer SADM 设置
    # =====================================================
    "use_learnable_layer_weight": True,

    # 注意：只有 [B, C, N] 的逐点层适合做 SADM
    # x_gf / f1 / f2 / f3 是 [B, C]，不适合 sort(dim=2)
    "sadm_layers": ["x_m", "x_2", "x_1"],

    # base_coef 是相对尺度校准项
    # softmax weight 是自适应门控权重
    "sadm_base_coef": {
        "x_m": 0.2,
        "x_2": 0.05,
        "x_1": 0.01,
    },

    # softmax 可学习权重的初始化先验
    "sadm_weight_prior": {
        "x_m": 0.75,
        "x_2": 0.20,
        "x_1": 0.05,
    },

    # layer weight logits 的学习率，不能和 lr_img 一样大
    "lr_layer_weight": 0.01,

    # KL 正则，防止权重太快塌缩到单层
    "layer_weight_kl": 0.001,

    "message": "AdaptiveMultiSADM_with_contribution_log",
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


def print_train_progress(it, total_it, loss_avg):
    percent = 100.0 * it / max(total_it, 1)
    print(f"[{percent:6.2f}%] Iter {it:04d}/{total_it} | loss = {loss_avg:.4f}")


# =========================================================
# Adaptive Multi-layer SADM 工具函数
# =========================================================
def sadm_layer_loss(feat_real, feat_syn):
    """
    feat_real / feat_syn: [B, C, N]

    原始 SADM:
    1. 每个 channel 沿点维度 N 排序；
    2. 对 batch 求平均，得到 [C, N]；
    3. 保持原始 SADM 的 channel-sum 写法：
       (((real - syn) ** 2).sum(dim=0)).mean()
    """
    sorted_real = torch.sort(feat_real, dim=2, descending=True)[0].detach()
    sorted_syn = torch.sort(feat_syn, dim=2, descending=True)[0]

    real = sorted_real.mean(dim=0)  # [C, N]
    syn = sorted_syn.mean(dim=0)    # [C, N]

    return (((real - syn) ** 2).sum(dim=0)).mean()


def weighted_sadm_loss(
    layers_real,
    layers_syn,
    layer_names,
    base_coef_dict,
    weight_logits,
):
    """
    对多个逐点层计算 SADM，并使用 learnable softmax 权重融合。

    返回：
    - total: 总 SADM loss
    - weights: softmax 后的层门控权重
    - raw_loss_items: 每层原始 SADM loss
    - contrib_items: 每层实际贡献
        contrib_i = weight_i * base_coef_i * raw_loss_i * len(layer_names)

    注意：
    contrib_items 和 total 的尺度一致，方便解释每层在总 SADM loss 中的实际贡献。
    """
    weights = torch.softmax(weight_logits, dim=0)

    total = torch.tensor(0.0, device=weight_logits.device)
    raw_loss_items = {}
    contrib_items = {}

    num_layers = len(layer_names)

    for i, name in enumerate(layer_names):
        layer_loss = sadm_layer_loss(layers_real[name], layers_syn[name])
        base_coef = base_coef_dict[name]

        # 未乘 num_layers 前的贡献
        layer_contrib = weights[i] * base_coef * layer_loss

        total = total + layer_contrib

        raw_loss_items[name] = layer_loss.detach()

        # 记录和 total 同尺度的实际贡献
        contrib_items[name] = (layer_contrib * num_layers).detach()

    # 乘层数，避免 softmax 融合后整体 loss 量级明显缩小
    total = total * num_layers

    return total, weights, raw_loss_items, contrib_items


def layer_weight_kl_loss(weight_logits, prior_values):
    """
    KL(w || prior)，防止 softmax 权重过早塌缩到单一层。
    如果不想限制权重，可以把 CONFIG["layer_weight_kl"] 设为 0.0。
    """
    eps = 1e-12
    weights = torch.softmax(weight_logits, dim=0)

    return torch.sum(
        weights * (
            torch.log(weights + eps) - torch.log(prior_values + eps)
        )
    )


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
    logger.info("Experiment: AdaSADM with layer contribution logging")

    # =========================================================
    # 数据集读取
    # =========================================================
    npoints, coord_dim, num_classes, dst_train, _, testloader = get_dataset(args, args.dataset)

    args.num_classes = num_classes
    model_eval_pool = get_eval_pool(args.eval_mode, args.model)

    # =========================================================
    # 评估轮次
    # =========================================================
    if args.TEST_ONLY:
        eval_it_pool = [args.TEST_EVAL_IT]
    else:
        if args.multieval:
            eval_it_pool = np.arange(0, args.Iteration + 1, 250).tolist()
            if args.Iteration not in eval_it_pool:
                eval_it_pool.append(args.Iteration)
        else:
            eval_it_pool = [args.Iteration]

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

        theta_x = torch.zeros(
            num_classes * args.ppc,
            dtype=torch.float,
            device=args.device
        ).requires_grad_(True)

        theta_y = torch.zeros(
            num_classes * args.ppc,
            dtype=torch.float,
            device=args.device
        ).requires_grad_(True)

        theta_z = torch.zeros(
            num_classes * args.ppc,
            dtype=torch.float,
            device=args.device
        ).requires_grad_(True)

        # =========================================================
        # Learnable layer weights
        # =========================================================
        sadm_layer_names = list(args.sadm_layers)

        prior_values = torch.tensor(
            [args.sadm_weight_prior[name] for name in sadm_layer_names],
            dtype=torch.float,
            device=args.device
        )
        prior_values = prior_values / prior_values.sum()

        # 用 log(prior) 初始化 logits，这样初始 softmax 权重就是 prior
        layer_weight_logits = torch.log(
            prior_values + 1e-12
        ).detach().clone().requires_grad_(True)

        logger.info("SADM layers: %s", sadm_layer_names)
        logger.info("SADM base coef: %s", args.sadm_base_coef)
        logger.info("Initial SADM layer prior: %s", prior_values.detach().cpu().numpy())

        optimizer_theta = torch.optim.SGD([
            {"params": theta_x, "lr": args.lr_rot / 10},
            {"params": theta_y, "lr": args.lr_rot},
            {"params": theta_z, "lr": args.lr_rot / 10},
        ], momentum=0.5)

        if args.init == "real":
            logger.info("Initialize synthetic data from random real samples")
            for c in range(num_classes):
                pointcloud_tmp.data[c * args.ppc:(c + 1) * args.ppc] = (
                    get_pointclouds(c, args.ppc).detach().data
                )
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
                pc_per_class = pointcloud_tmp.data[
                    c * args.ppc:(c + 1) * args.ppc
                ].cpu().numpy()
                save_pointcloud_txt_batch(pc_per_class, label_folder, f"init_{c}")

        # =========================================================
        # 训练准备
        # =========================================================
        optimizer_img = torch.optim.SGD(
            [pointcloud_tmp],
            lr=args.lr_img,
            momentum=0.5
        )

        optimizer_layer_weight = torch.optim.Adam(
            [layer_weight_logits],
            lr=args.lr_layer_weight
        )

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

                        eval_seed = 1996 + it_eval
                        random.seed(eval_seed)
                        np.random.seed(eval_seed)
                        torch.manual_seed(eval_seed)
                        torch.cuda.manual_seed(eval_seed)
                        torch.cuda.manual_seed_all(eval_seed)

                        net_eval = get_network(
                            model_eval,
                            channel,
                            num_classes,
                            args.feature_transform
                        ).to(args.device)

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
                    logger.info(
                        "Model %s | mean acc = %.4f | std = %.4f",
                        model_eval,
                        mean_acc,
                        std_acc
                    )

                    accs_all.append(mean_acc)

                    if it == args.Iteration or args.TEST_ONLY:
                        accs_all_exps[model_eval] += accs

                if args.SAVE_SYNTHETIC_TXT:
                    pointcloud_syn_vis = copy.deepcopy(pointcloud_syn.detach().cpu().numpy())
                    for c in range(num_classes):
                        label_folder = os.path.join(pc_div_name, f"class_{c}")
                        os.makedirs(label_folder, exist_ok=True)
                        pc_syn_per_class = pointcloud_syn_vis[
                            c * args.ppc:(c + 1) * args.ppc
                        ]
                        save_pointcloud_txt_batch(
                            pc_syn_per_class,
                            label_folder,
                            f"iter_{it}_class_{c}"
                        )

                if args.TEST_ONLY:
                    logger.info("TEST_ONLY=True, stop after first evaluation.")
                    break

            # =====================================================
            # 更新 synthetic 数据
            # =====================================================
            net = get_network(
                args.model,
                channel,
                num_classes,
                args.feature_transform
            ).to(args.device)

            net.train()

            for param in net.parameters():
                param.requires_grad = False

            bn_flag = any("BatchNorm" in module._get_name() for module in net.modules())
            if bn_flag:
                bn_size_pc = 8
                pc_real = torch.cat(
                    [get_pointclouds(c, bn_size_pc) for c in range(num_classes)],
                    dim=0
                )
                net.train()
                _ = net(pc_real)
                for module in net.modules():
                    if "BatchNorm" in module._get_name():
                        module.eval()

            loss = torch.tensor(0.0, device=args.device)

            # 用来记录每个 class 的 raw loss / contribution
            raw_loss_tracker = {
                name: []
                for name in sadm_layer_names
            }
            contrib_tracker = {
                name: []
                for name in sadm_layer_names
            }

            for c in range(num_classes):
                pc_syn = pointcloud_syn[c * args.ppc:(c + 1) * args.ppc].reshape(
                    args.ppc,
                    coord_dim,
                    npoints
                ).to(args.device)

                pc_real = get_pointclouds(c, args.batch_real)

                with torch.no_grad():
                    _, _, _, _, layers_real = net(pc_real)

                _, _, _, _, layers_syn = net(pc_syn)

                # =================================================
                # Adaptive Multi-layer SADM
                # =================================================
                loss_sadm_weighted, sadm_weights, sadm_loss_items, sadm_contrib_items = weighted_sadm_loss(
                    layers_real=layers_real,
                    layers_syn=layers_syn,
                    layer_names=sadm_layer_names,
                    base_coef_dict=args.sadm_base_coef,
                    weight_logits=layer_weight_logits,
                )

                for name in sadm_layer_names:
                    raw_loss_tracker[name].append(sadm_loss_items[name])
                    contrib_tracker[name].append(sadm_contrib_items[name])

                # x_gf 仍然使用 M3D
                loss_m3d = m3d_criterion(
                    layers_real["x_gf"],
                    layers_syn["x_gf"]
                )

                loss1 = loss_sadm_weighted
                loss1 += loss_m3d * 0.001

                loss += loss1 * args.ppc

            # KL 正则只加一次，不要在每个 class 里面重复加
            if args.use_learnable_layer_weight and args.layer_weight_kl > 0:
                loss += args.layer_weight_kl * layer_weight_kl_loss(
                    layer_weight_logits,
                    prior_values
                )

            optimizer_img.zero_grad()
            optimizer_theta.zero_grad()
            optimizer_layer_weight.zero_grad()

            loss.backward()

            optimizer_img.step()
            optimizer_theta.step()
            optimizer_layer_weight.step()

            loss_avg = loss.item() / num_classes

            if (it % args.PRINT_EVERY == 0) or (it == args.Iteration):
                print_train_progress(it, args.Iteration, loss_avg)

            # =====================================================
            # 打印当前可学习层权重 + raw loss + effective contribution
            # =====================================================
            if (it % (args.PRINT_EVERY * 10) == 0) or (it == args.Iteration):
                with torch.no_grad():
                    w = torch.softmax(layer_weight_logits, dim=0).detach().cpu().numpy()

                    raw_mean = {}
                    contrib_mean = {}

                    for name in sadm_layer_names:
                        raw_mean[name] = torch.stack(raw_loss_tracker[name]).mean().item()
                        contrib_mean[name] = torch.stack(contrib_tracker[name]).mean().item()

                    weight_msg = " | ".join([
                        f"{name}={w_i:.4f}"
                        for name, w_i in zip(sadm_layer_names, w)
                    ])

                    raw_msg = " | ".join([
                        f"{name}_raw={raw_mean[name]:.6f}"
                        for name in sadm_layer_names
                    ])

                    contrib_msg = " | ".join([
                        f"{name}_contrib={contrib_mean[name]:.6f}"
                        for name in sadm_layer_names
                    ])

                    contrib_sum = sum(contrib_mean.values()) + 1e-12
                    contrib_ratio_msg = " | ".join([
                        f"{name}_ratio={contrib_mean[name] / contrib_sum:.4f}"
                        for name in sadm_layer_names
                    ])

                    logger.info("[Layer Weights] Iter %d | %s", it, weight_msg)
                    logger.info("[Layer Raw Loss] Iter %d | %s", it, raw_msg)
                    logger.info("[Layer Contribution] Iter %d | %s", it, contrib_msg)
                    logger.info("[Layer Contribution Ratio] Iter %d | %s", it, contrib_ratio_msg)

                    print(f"[Layer Weights] Iter {it} | {weight_msg}")
                    print(f"[Layer Raw Loss] Iter {it} | {raw_msg}")
                    print(f"[Layer Contribution] Iter {it} | {contrib_msg}")
                    print(f"[Layer Contribution Ratio] Iter {it} | {contrib_ratio_msg}")

            # =====================================================
            # 保存最终 synthetic 数据
            # =====================================================
            if it == args.Iteration:
                data_save.append([
                    copy.deepcopy(pointcloud_syn.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu())
                ])

                save_dict = {
                    "data": data_save,
                    "accs_all_exps": accs_all_exps,
                    "sadm_layer_names": sadm_layer_names,
                    "sadm_layer_weights": torch.softmax(
                        layer_weight_logits,
                        dim=0
                    ).detach().cpu(),
                    "sadm_weight_logits": layer_weight_logits.detach().cpu(),
                    "sadm_weight_prior": prior_values.detach().cpu(),
                    "sadm_base_coef": args.sadm_base_coef,
                    "experiment_type": "adasadm_with_contribution_log",
                }

                save_file = os.path.join(
                    args.save_path,
                    f"res_{args.method}_{args.dataset}_{args.model}_{args.ppc}ppc_{args.message}.pt"
                )

                torch.save(save_dict, save_file)
                logger.info("Saved result to %s", save_file)

    # =========================================================
    # 打印最终结果
    # =========================================================
    logger.info("\n==================== Final Results ====================\n")
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        if len(accs) > 0:
            logger.info(
                "Run %d exp | train on %s | eval %d random %s | mean=%.2f%% std=%.2f%%",
                args.num_exp,
                args.model,
                len(accs),
                key,
                np.mean(accs) * 100,
                np.std(accs) * 100
            )

    if len(accs_all) > 0:
        print("\n==================== Final Results ====================\n")
        print(np.array(accs_all))


if __name__ == "__main__":
    set_seed(1999)
    main()