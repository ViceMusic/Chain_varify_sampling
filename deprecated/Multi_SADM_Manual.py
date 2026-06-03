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
    # Replay Schedule SADM
    # =====================================================
    "sadm_layers": ["x_m", "x_2", "x_1"],

    # base_coef 是相对尺度校准项，和 AdaSADM 保持一致
    "sadm_base_coef": {
        "x_m": 0.2,
        "x_2": 0.05,
        "x_1": 0.01,
    },

    # =====================================================
    # 从 AdaSADM 日志中记录下来的 learned gate trajectory
    # 每 100 iter 一个点
    # =====================================================
    "replay_weight_schedule": [
        (0,    {"x_m": 0.7462, "x_2": 0.2030, "x_1": 0.0508}),
        (100,  {"x_m": 0.3146, "x_2": 0.5468, "x_1": 0.1386}),
        (200,  {"x_m": 0.1066, "x_2": 0.6974, "x_1": 0.1960}),
        (300,  {"x_m": 0.0535, "x_2": 0.6974, "x_1": 0.2492}),
        (400,  {"x_m": 0.0324, "x_2": 0.6337, "x_1": 0.3340}),
        (500,  {"x_m": 0.0208, "x_2": 0.5118, "x_1": 0.4674}),
        (600,  {"x_m": 0.0132, "x_2": 0.3594, "x_1": 0.6274}),
        (700,  {"x_m": 0.0084, "x_2": 0.2335, "x_1": 0.7581}),
        (800,  {"x_m": 0.0055, "x_2": 0.1511, "x_1": 0.8434}),
        (900,  {"x_m": 0.0039, "x_2": 0.1077, "x_1": 0.8884}),
        (1000, {"x_m": 0.0029, "x_2": 0.0790, "x_1": 0.9180}),
        (1100, {"x_m": 0.0023, "x_2": 0.0606, "x_1": 0.9372}),
        (1200, {"x_m": 0.0019, "x_2": 0.0488, "x_1": 0.9493}),
        (1300, {"x_m": 0.0015, "x_2": 0.0401, "x_1": 0.9584}),
        (1400, {"x_m": 0.0013, "x_2": 0.0333, "x_1": 0.9654}),
        (1500, {"x_m": 0.0011, "x_2": 0.0285, "x_1": 0.9704}),
        (1600, {"x_m": 0.0010, "x_2": 0.0243, "x_1": 0.9747}),
        (1700, {"x_m": 0.0008, "x_2": 0.0212, "x_1": 0.9780}),
        (1800, {"x_m": 0.0007, "x_2": 0.0185, "x_1": 0.9808}),
        (1900, {"x_m": 0.0007, "x_2": 0.0163, "x_1": 0.9830}),
        (2000, {"x_m": 0.0006, "x_2": 0.0145, "x_1": 0.9849}),
        (2100, {"x_m": 0.0005, "x_2": 0.0130, "x_1": 0.9865}),
        (2200, {"x_m": 0.0005, "x_2": 0.0116, "x_1": 0.9879}),
        (2300, {"x_m": 0.0004, "x_2": 0.0105, "x_1": 0.9891}),
        (2400, {"x_m": 0.0004, "x_2": 0.0095, "x_1": 0.9901}),
        (2500, {"x_m": 0.0004, "x_2": 0.0087, "x_1": 0.9910}),
        (2600, {"x_m": 0.0003, "x_2": 0.0079, "x_1": 0.9918}),
        (2700, {"x_m": 0.0003, "x_2": 0.0073, "x_1": 0.9924}),
        (2800, {"x_m": 0.0003, "x_2": 0.0067, "x_1": 0.9931}),
        (2900, {"x_m": 0.0003, "x_2": 0.0062, "x_1": 0.9936}),
        (3000, {"x_m": 0.0002, "x_2": 0.0057, "x_1": 0.9941}),
        (3100, {"x_m": 0.0002, "x_2": 0.0052, "x_1": 0.9945}),
        (3200, {"x_m": 0.0002, "x_2": 0.0049, "x_1": 0.9949}),
        (3300, {"x_m": 0.0002, "x_2": 0.0045, "x_1": 0.9953}),
        (3400, {"x_m": 0.0002, "x_2": 0.0042, "x_1": 0.9956}),
        (3500, {"x_m": 0.0002, "x_2": 0.0039, "x_1": 0.9959}),
        (3600, {"x_m": 0.0002, "x_2": 0.0036, "x_1": 0.9962}),
        (3700, {"x_m": 0.0001, "x_2": 0.0034, "x_1": 0.9965}),
        (3800, {"x_m": 0.0001, "x_2": 0.0032, "x_1": 0.9967}),
        (3900, {"x_m": 0.0001, "x_2": 0.0030, "x_1": 0.9969}),
        (4000, {"x_m": 0.0001, "x_2": 0.0028, "x_1": 0.9971}),
    ],

    # True：相邻 100 iter 之间线性插值，更接近连续变化
    # False：每 100 iter 阶梯固定
    "replay_interpolate": True,

    "message": "ReplayScheduleSADM_100iter_interp",
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
# SADM 工具函数
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

    real = sorted_real.mean(dim=0)
    syn = sorted_syn.mean(dim=0)

    return (((real - syn) ** 2).sum(dim=0)).mean()


def get_replay_layer_weights(args, layer_names, it):
    """
    根据 AdaSADM 记录下来的权重轨迹，在当前 iter 获取 replay 权重。

    replay_interpolate=True:
        相邻记录点之间线性插值。

    replay_interpolate=False:
        阶梯函数，每 100 iter 使用左端点权重。
    """
    schedule = sorted(args.replay_weight_schedule, key=lambda x: x[0])

    # it 小于第一个点
    if it <= schedule[0][0]:
        weight_dict = schedule[0][1]
        weights = torch.tensor(
            [weight_dict[name] for name in layer_names],
            dtype=torch.float,
            device=args.device
        )
        return weights / weights.sum(), f"replay_{schedule[0][0]}"

    # it 大于最后一个点
    if it >= schedule[-1][0]:
        weight_dict = schedule[-1][1]
        weights = torch.tensor(
            [weight_dict[name] for name in layer_names],
            dtype=torch.float,
            device=args.device
        )
        return weights / weights.sum(), f"replay_{schedule[-1][0]}"

    # 找到 it 所在区间 [left_it, right_it]
    for idx in range(len(schedule) - 1):
        left_it, left_w = schedule[idx]
        right_it, right_w = schedule[idx + 1]

        if left_it <= it < right_it:
            if args.replay_interpolate:
                ratio = (it - left_it) / float(right_it - left_it)

                interp_w = {}
                for name in layer_names:
                    interp_w[name] = (
                        (1.0 - ratio) * left_w[name]
                        + ratio * right_w[name]
                    )

                weights = torch.tensor(
                    [interp_w[name] for name in layer_names],
                    dtype=torch.float,
                    device=args.device
                )

                return weights / weights.sum(), f"interp_{left_it}_{right_it}"

            else:
                weights = torch.tensor(
                    [left_w[name] for name in layer_names],
                    dtype=torch.float,
                    device=args.device
                )

                return weights / weights.sum(), f"step_{left_it}"

    # 兜底
    weight_dict = schedule[-1][1]
    weights = torch.tensor(
        [weight_dict[name] for name in layer_names],
        dtype=torch.float,
        device=args.device
    )
    return weights / weights.sum(), f"replay_{schedule[-1][0]}"


def scheduled_sadm_loss(
    layers_real,
    layers_syn,
    layer_names,
    base_coef_dict,
    schedule_weights,
    device,
):
    """
    使用 replay schedule 权重计算多层 SADM。

    loss = len(layers) * Σ_i schedule_w_i * base_coef_i * L_i

    schedule_w_i:
        从 AdaSADM 权重轨迹中 replay 出来的权重。

    base_coef_i:
        相对尺度校准项，和 AdaSADM 保持一致。
    """
    total = torch.tensor(0.0, device=device)
    loss_items = {}

    for i, name in enumerate(layer_names):
        layer_loss = sadm_layer_loss(layers_real[name], layers_syn[name])
        base_coef = base_coef_dict[name]

        total = total + schedule_weights[i] * base_coef * layer_loss
        loss_items[name] = layer_loss.detach()

    # 和 AdaSADM 保持一致：加权后乘层数
    total = total * len(layer_names)

    return total, loss_items


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
    logger.info("Experiment: Replay Schedule SADM")
    logger.info("Replay interpolate: %s", args.replay_interpolate)

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
        # SADM 层设置
        # =========================================================
        sadm_layer_names = list(args.sadm_layers)

        logger.info("SADM layers: %s", sadm_layer_names)
        logger.info("SADM base coef: %s", args.sadm_base_coef)

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

            # 当前 replay schedule 权重
            schedule_weights, schedule_name = get_replay_layer_weights(
                args,
                sadm_layer_names,
                it
            )

            loss = torch.tensor(0.0, device=args.device)

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

                loss_sadm_schedule, sadm_loss_items = scheduled_sadm_loss(
                    layers_real=layers_real,
                    layers_syn=layers_syn,
                    layer_names=sadm_layer_names,
                    base_coef_dict=args.sadm_base_coef,
                    schedule_weights=schedule_weights,
                    device=args.device,
                )

                loss_m3d = m3d_criterion(
                    layers_real["x_gf"],
                    layers_syn["x_gf"]
                )

                loss1 = loss_sadm_schedule
                loss1 += loss_m3d * 0.001

                loss += loss1 * args.ppc

            optimizer_img.zero_grad()
            optimizer_theta.zero_grad()

            loss.backward()

            optimizer_img.step()
            optimizer_theta.step()

            loss_avg = loss.item() / num_classes

            if (it % args.PRINT_EVERY == 0) or (it == args.Iteration):
                print_train_progress(it, args.Iteration, loss_avg)

            # 打印当前 replay 权重
            if (it % (args.PRINT_EVERY * 10) == 0) or (it == args.Iteration):
                with torch.no_grad():
                    w = schedule_weights.detach().cpu().numpy()
                    weight_msg = " | ".join([
                        f"{name}={w_i:.4f}"
                        for name, w_i in zip(sadm_layer_names, w)
                    ])

                    logger.info(
                        "[Replay Schedule Weights] Iter %d | schedule=%s | %s",
                        it,
                        schedule_name,
                        weight_msg
                    )
                    print(
                        f"[Replay Schedule Weights] Iter {it} | "
                        f"schedule={schedule_name} | {weight_msg}"
                    )

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
                    "sadm_base_coef": args.sadm_base_coef,
                    "replay_weight_schedule": args.replay_weight_schedule,
                    "replay_interpolate": args.replay_interpolate,
                    "experiment_type": "replay_schedule_sadm",
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