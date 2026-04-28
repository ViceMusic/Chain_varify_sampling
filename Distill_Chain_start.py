import os
import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import torch

from utils import *


# Base config. Copied from start_template.py; only the distillation loss is replaced.
CONFIG = {
    "method": "AnchorChain",
    "dataset": "MODELNET40_H5",
    "model": "PointNet",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # distillation
    "ppc": 10,
    "eval_mode": "S",
    "Multi":False,
    "num_exp": 1,
    "num_eval": 10,
    "epoch_eval_train": 500,
    "Iteration": 1500,
    "eval_interval": 250,

    "lr_img": 10.0,
    "lr_net": 0.01,
    "lr_rot": 0.01,
    "batch_real": 8,
    "batch_train": 8,
    "init": "real",
    "feature_transform": 0,

    # learnable rotation alignment
    "use_rotation": True,

    # AnchorChain loss
    "anchor_k": 64,
    "kmedoids_iters": 4,
    "sinkhorn_iters": 20,
    "sinkhorn_tau": 0.08,
    "chain_eps": 1e-6,
    "lambda_geo": 1.0,
    "lambda_feat": 1.0,
    "lambda_delta": 0.5,
    "lambda_ratio": 0.1,

    # save / print
    "save_path": "result_h5",
    "mode": "result_h5",
    "PRINT_EVERY": 10,
}

# 把上面的内容当作一个整体，转换成一个对象，方便后续调用
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
    """
    根据每个 synthetic 样本的旋转角，构造 batch 形式的 3D 旋转矩阵。

    angle: [B]
    return: [B, 3, 3]
    """
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

    elif rot_type == "z":
        return torch.stack([
            torch.stack([c, -s, zeros], dim=1),
            torch.stack([s, c, zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=2)

    else:
        raise ValueError(f"Unknown rot_type: {rot_type}")

def print_train_progress(it, total_it, loss_avg):
    percent = 100.0 * it / max(total_it, 1)
    print(f"[{percent:6.2f}%] Iter {it:04d}/{total_it} | anchor_chain_loss = {loss_avg:.4f}")

def evaluate_current_synset(pointcloud_syn, label_syn, testloader, args, channel, num_classes):
    """Evaluate current synthetic set multiple times and return mean/std/best-run result."""
    model_eval_pool = get_eval_pool(args.eval_mode, args.model)

    eval_summary = {}
    best_mean_acc = -1.0
    best_model_name = None

    for model_eval in model_eval_pool:
        accs = []
        best_single_acc = -1.0
        best_single_per_class = None

        for it_eval in range(args.num_eval):
            torch.manual_seed(1996 + it_eval)

            net_eval = get_network(model_eval, channel, num_classes, args.feature_transform).to(args.device)
            syn_eval = copy.deepcopy(pointcloud_syn.detach())
            label_eval = copy.deepcopy(label_syn.detach())

            _, _, acc_test, acc_test_per_class, _ = evaluate_synset(
                it_eval,
                net_eval,
                syn_eval,
                label_eval,
                testloader,
                args
            )
            accs.append(acc_test)

            if acc_test > best_single_acc:
                best_single_acc = acc_test
                best_single_per_class = acc_test_per_class

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))

        eval_summary[model_eval] = {
            "accs": accs,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "best_single_acc": best_single_acc,
            "best_single_per_class": best_single_per_class,
        }

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_model_name = model_eval

    return eval_summary, best_model_name, best_mean_acc


def main():
    args = build_args_from_config(CONFIG)
    os.makedirs(args.mode, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    print("========== Basic Info ==========")
    print(f"Device   : {args.device}")
    print(f"Dataset  : {args.dataset}")
    print(f"Model    : {args.model}")
    print(f"Iter     : {args.Iteration}")
    print(f"PPC      : {args.ppc}")
    print(f"Anchors  : {args.anchor_k}")
    print("================================")

    # dataset: follow start_old.py, no single-file fallback in the main distillation flow.
    npoints, coord_dim, num_classes, dst_train, _, testloader = get_dataset(args, args.dataset)
    args.num_classes = num_classes
    channel = 3

    # fixed evaluation schedule
    eval_it_pool = list(range(0, args.Iteration + 1, args.eval_interval)) if args.Multi else [args.Iteration + 1]
    if eval_it_pool[-1] != args.Iteration:
        eval_it_pool.append(args.Iteration)

    best_eval_acc = -1.0
    best_eval_iter = -1
    best_eval_model = None
    best_data_package = None
    final_eval_summary = None

    for exp in range(args.num_exp):
        print(f"\n========== Experiment {exp + 1}/{args.num_exp} ==========")

        # organize real data
        pointcloud_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))]
        indices_class = [[] for _ in range(num_classes)]

        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        pointcloud_all = torch.cat(pointcloud_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        def get_pointclouds(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return pointcloud_all[idx_shuffle]

        # init synthetic data
        pointcloud_tmp = torch.randn(
            size=(num_classes * args.ppc, coord_dim, npoints),
            dtype=torch.float,
            requires_grad=True,
            device=args.device
        )
        pointcloud_tmp = torch.tanh(pointcloud_tmp).detach().clone().requires_grad_(True)

        if args.init == "real":
            for c in range(num_classes):
                pointcloud_tmp.data[c * args.ppc:(c + 1) * args.ppc] = get_pointclouds(c, args.ppc).detach().data

        label_syn = torch.tensor(
            [np.ones(args.ppc) * i for i in range(num_classes)],
            dtype=torch.long,
            requires_grad=False,
            device=args.device
        ).view(-1)

        optimizer_img = torch.optim.SGD([pointcloud_tmp], lr=args.lr_img, momentum=0.5)

        # learnable rotation parameters for each synthetic sample
        num_syn = num_classes * args.ppc
        theta_x = torch.zeros(num_syn, dtype=torch.float, device=args.device, requires_grad=True)
        theta_y = torch.zeros(num_syn, dtype=torch.float, device=args.device, requires_grad=True)
        theta_z = torch.zeros(num_syn, dtype=torch.float, device=args.device, requires_grad=True)

        optimizer_rot = torch.optim.SGD([
            {"params": theta_x, "lr": args.lr_rot / 10},
            {"params": theta_y, "lr": args.lr_rot},
            {"params": theta_z, "lr": args.lr_rot / 10},
        ], momentum=0.5)

        # distillation loop
        for it in range(args.Iteration + 1):
            # Normalize synthetic point clouds first.
            # pointcloud_tmp: raw learnable synthetic coordinates
            # pointcloud_syn_base: normalized synthetic point clouds before rotation
            pointcloud_syn_base = pc_normalize_batch(pointcloud_tmp)

            if args.use_rotation:
                # keep rotation angles in [-pi, pi] for numerical stability
                with torch.no_grad():
                    theta_x.data = torch.remainder(theta_x.data + math.pi, 2 * math.pi) - math.pi
                    theta_y.data = torch.remainder(theta_y.data + math.pi, 2 * math.pi) - math.pi
                    theta_z.data = torch.remainder(theta_z.data + math.pi, 2 * math.pi) - math.pi

                rot_matrix_x = create_rotation_matrix(theta_x, "x")
                rot_matrix_y = create_rotation_matrix(theta_y, "y")
                rot_matrix_z = create_rotation_matrix(theta_z, "z")

                # combined rotation matrix for each synthetic sample
                rot_matrix = torch.bmm(rot_matrix_z, torch.bmm(rot_matrix_y, rot_matrix_x))

                # [B, 3, N] -> [B, N, 3] -> rotate -> [B, 3, N]
                pointcloud_syn_xyz = pointcloud_syn_base.permute(0, 2, 1).contiguous()
                pointcloud_syn_rotated = torch.bmm(pointcloud_syn_xyz, rot_matrix)
                pointcloud_syn = pointcloud_syn_rotated.permute(0, 2, 1).contiguous()

                # optional: normalize again after rotation to avoid tiny numerical drift
                pointcloud_syn = pc_normalize_batch(pointcloud_syn)

            else:
                pointcloud_syn = pointcloud_syn_base

            # evaluate current synthetic set
            if it in eval_it_pool:
                eval_summary, best_model_name, current_eval_acc = evaluate_current_synset(
                    pointcloud_syn, label_syn, testloader, args, channel, num_classes
                )
                final_eval_summary = eval_summary

                print(f"[Eval] Iter {it:04d} | best mean acc = {current_eval_acc:.4f} | model = {best_model_name}")

                if current_eval_acc > best_eval_acc:
                    best_eval_acc = current_eval_acc
                    best_eval_iter = it
                    best_eval_model = best_model_name
                    best_data_package = best_data_package = {
                        "pointcloud_syn": copy.deepcopy(pointcloud_syn.detach().cpu()),
                        "pointcloud_tmp": copy.deepcopy(pointcloud_tmp.detach().cpu()),
                        "label_syn": copy.deepcopy(label_syn.detach().cpu()),
                        "theta_x": copy.deepcopy(theta_x.detach().cpu()) if args.use_rotation else None,
                        "theta_y": copy.deepcopy(theta_y.detach().cpu()) if args.use_rotation else None,
                        "theta_z": copy.deepcopy(theta_z.detach().cpu()) if args.use_rotation else None,
                        "eval_summary": copy.deepcopy(eval_summary),
                        "best_eval_acc": best_eval_acc,
                        "best_eval_iter": best_eval_iter,
                        "best_eval_model": best_eval_model,
                        "config": dict(CONFIG),
                    }
                    save_file = os.path.join(
                        args.save_path,
                        f"best_res_{args.method}_{args.dataset}_{args.model}_{args.ppc}ppc.pt"
                    )
                    torch.save(best_data_package, save_file)

            # frozen network for feature matching
            net = get_network(args.model, channel, num_classes, args.feature_transform).to(args.device)
            net.train()

            for param in net.parameters():
                param.requires_grad = False

            bn_flag = any("BatchNorm" in module._get_name() for module in net.modules())
            if bn_flag:
                bn_size_pc = 8
                pc_real_bn = torch.cat([get_pointclouds(c, bn_size_pc) for c in range(num_classes)], dim=0)
                _ = net(pc_real_bn)
                for module in net.modules():
                    if "BatchNorm" in module._get_name():
                        module.eval()

            # distillation loss
            loss = torch.tensor(0.0, device=args.device)

            for c in range(num_classes):
                pc_syn = pointcloud_syn[c * args.ppc:(c + 1) * args.ppc].reshape(
                    args.ppc, coord_dim, npoints
                ).to(args.device)
                pc_real = get_pointclouds(c, args.batch_real)
                with torch.no_grad():
                    _, _, _, _, layers_real = net(pc_real)
                _, _, _, _, layers_syn = net(pc_syn)

                # AnchorChain:
                # 1. real/syn 分别用 k-medoids 选 K 个 anchor；
                # 2. 用 encoder 的 x_m anchor feature 构造 Sinkhorn 代价并硬化匹配；
                # 3. 按匹配后的顺序分别计算几何链和 feature 链；
                # 4. 对 geo/feat/delta/ratio 四个描述子做 L1 蒸馏。
                loss_c = torch.tensor(0.0, device=args.device)
                pair_count = min(pc_real.shape[0], pc_syn.shape[0])

                for pair_i in range(pair_count):
                    real_points = pc_real[pair_i].T.contiguous()
                    syn_points = pc_syn[pair_i].T.contiguous()
                    real_xm = layers_real["x_m"][pair_i].T.contiguous()
                    syn_xm = layers_syn["x_m"][pair_i].T.contiguous()

                    real_anchor_idx = torch.linspace(
                        0, real_points.shape[0] - 1, steps=args.anchor_k, device=args.device
                    ).long()
                    syn_anchor_idx = torch.linspace(
                        0, syn_points.shape[0] - 1, steps=args.anchor_k, device=args.device
                    ).long()

                    for _ in range(args.kmedoids_iters):
                        real_dist = torch.cdist(real_points, real_points[real_anchor_idx])
                        syn_dist = torch.cdist(syn_points, syn_points[syn_anchor_idx])
                        real_assign = real_dist.argmin(dim=1)
                        syn_assign = syn_dist.argmin(dim=1)
                        new_real_idx = real_anchor_idx.clone()
                        new_syn_idx = syn_anchor_idx.clone()

                        for anchor_i in range(args.anchor_k):
                            real_members_idx = torch.where(real_assign == anchor_i)[0]
                            if real_members_idx.numel() > 0:
                                real_members = real_points[real_members_idx]
                                real_inner_dist = torch.cdist(real_members, real_members)
                                new_real_idx[anchor_i] = real_members_idx[real_inner_dist.sum(dim=1).argmin()]

                            syn_members_idx = torch.where(syn_assign == anchor_i)[0]
                            if syn_members_idx.numel() > 0:
                                syn_members = syn_points[syn_members_idx]
                                syn_inner_dist = torch.cdist(syn_members, syn_members)
                                new_syn_idx[anchor_i] = syn_members_idx[syn_inner_dist.sum(dim=1).argmin()]

                        if torch.equal(new_real_idx, real_anchor_idx) and torch.equal(new_syn_idx, syn_anchor_idx):
                            break
                        real_anchor_idx = new_real_idx
                        syn_anchor_idx = new_syn_idx

                    real_anchor = real_points[real_anchor_idx]
                    syn_anchor = syn_points[syn_anchor_idx]
                    real_anchor_feat = real_xm[real_anchor_idx]
                    syn_anchor_feat = syn_xm[syn_anchor_idx]

                    cost = torch.cdist(real_anchor_feat.detach(), syn_anchor_feat.detach())
                    transport = torch.exp(-cost / args.sinkhorn_tau)
                    for _ in range(args.sinkhorn_iters):
                        transport = transport / (transport.sum(dim=1, keepdim=True) + args.chain_eps)
                        transport = transport / (transport.sum(dim=0, keepdim=True) + args.chain_eps)

                    used_syn_anchor = torch.zeros(args.anchor_k, dtype=torch.bool, device=args.device)
                    match_idx = torch.zeros(args.anchor_k, dtype=torch.long, device=args.device)
                    for anchor_i in range(args.anchor_k):
                        row_order = torch.argsort(transport[anchor_i], descending=True)
                        chosen = row_order[0]
                        for candidate in row_order:
                            if not used_syn_anchor[candidate]:
                                chosen = candidate
                                break
                        match_idx[anchor_i] = chosen
                        used_syn_anchor[chosen] = True

                    syn_anchor = syn_anchor[match_idx]
                    syn_anchor_feat = syn_anchor_feat[match_idx]

                    real_vec = real_anchor[1:] - real_anchor[:-1]
                    syn_vec = syn_anchor[1:] - syn_anchor[:-1]
                    real_r = torch.norm(real_vec, dim=1)
                    syn_r = torch.norm(syn_vec, dim=1)
                    real_theta = torch.acos(torch.clamp(
                        (real_vec[:-1] * real_vec[1:]).sum(dim=1)
                        / (torch.norm(real_vec[:-1], dim=1) * torch.norm(real_vec[1:], dim=1) + args.chain_eps),
                        -1.0, 1.0
                    ))
                    syn_theta = torch.acos(torch.clamp(
                        (syn_vec[:-1] * syn_vec[1:]).sum(dim=1)
                        / (torch.norm(syn_vec[:-1], dim=1) * torch.norm(syn_vec[1:], dim=1) + args.chain_eps),
                        -1.0, 1.0
                    ))
                    real_gr = torch.norm(real_anchor[2:] - real_anchor[:-2], dim=1)
                    syn_gr = torch.norm(syn_anchor[2:] - syn_anchor[:-2], dim=1)
                    real_gtheta = torch.acos(torch.clamp(
                        ((real_anchor[1:-1] - real_anchor[:-2]) * (real_anchor[2:] - real_anchor[:-2])).sum(dim=1)
                        / (
                            torch.norm(real_anchor[1:-1] - real_anchor[:-2], dim=1)
                            * torch.norm(real_anchor[2:] - real_anchor[:-2], dim=1)
                            + args.chain_eps
                        ),
                        -1.0, 1.0
                    ))
                    syn_gtheta = torch.acos(torch.clamp(
                        ((syn_anchor[1:-1] - syn_anchor[:-2]) * (syn_anchor[2:] - syn_anchor[:-2])).sum(dim=1)
                        / (
                            torch.norm(syn_anchor[1:-1] - syn_anchor[:-2], dim=1)
                            * torch.norm(syn_anchor[2:] - syn_anchor[:-2], dim=1)
                            + args.chain_eps
                        ),
                        -1.0, 1.0
                    ))

                    real_fvec = real_anchor_feat[1:] - real_anchor_feat[:-1]
                    syn_fvec = syn_anchor_feat[1:] - syn_anchor_feat[:-1]
                    real_fr = torch.norm(real_fvec, dim=1)
                    syn_fr = torch.norm(syn_fvec, dim=1)
                    real_ftheta = torch.acos(torch.clamp(
                        (real_fvec[:-1] * real_fvec[1:]).sum(dim=1)
                        / (torch.norm(real_fvec[:-1], dim=1) * torch.norm(real_fvec[1:], dim=1) + args.chain_eps),
                        -1.0, 1.0
                    ))
                    syn_ftheta = torch.acos(torch.clamp(
                        (syn_fvec[:-1] * syn_fvec[1:]).sum(dim=1)
                        / (torch.norm(syn_fvec[:-1], dim=1) * torch.norm(syn_fvec[1:], dim=1) + args.chain_eps),
                        -1.0, 1.0
                    ))
                    real_fgr = torch.norm(real_anchor_feat[2:] - real_anchor_feat[:-2], dim=1)
                    syn_fgr = torch.norm(syn_anchor_feat[2:] - syn_anchor_feat[:-2], dim=1)
                    real_fgtheta = torch.acos(torch.clamp(
                        (
                            (real_anchor_feat[1:-1] - real_anchor_feat[:-2])
                            * (real_anchor_feat[2:] - real_anchor_feat[:-2])
                        ).sum(dim=1)
                        / (
                            torch.norm(real_anchor_feat[1:-1] - real_anchor_feat[:-2], dim=1)
                            * torch.norm(real_anchor_feat[2:] - real_anchor_feat[:-2], dim=1)
                            + args.chain_eps
                        ),
                        -1.0, 1.0
                    ))
                    syn_fgtheta = torch.acos(torch.clamp(
                        (
                            (syn_anchor_feat[1:-1] - syn_anchor_feat[:-2])
                            * (syn_anchor_feat[2:] - syn_anchor_feat[:-2])
                        ).sum(dim=1)
                        / (
                            torch.norm(syn_anchor_feat[1:-1] - syn_anchor_feat[:-2], dim=1)
                            * torch.norm(syn_anchor_feat[2:] - syn_anchor_feat[:-2], dim=1)
                            + args.chain_eps
                        ),
                        -1.0, 1.0
                    ))

                    real_geo = torch.cat([real_r, real_theta, real_gr, real_gtheta], dim=0)
                    syn_geo = torch.cat([syn_r, syn_theta, syn_gr, syn_gtheta], dim=0)
                    real_feat = torch.cat([real_fr, real_ftheta, real_fgr, real_fgtheta], dim=0)
                    syn_feat = torch.cat([syn_fr, syn_ftheta, syn_fgr, syn_fgtheta], dim=0)

                    real_delta = torch.abs(real_geo - real_feat)
                    syn_delta = torch.abs(syn_geo - syn_feat)
                    real_ratio = real_feat / (torch.abs(real_geo) + args.chain_eps)
                    syn_ratio = syn_feat / (torch.abs(syn_geo) + args.chain_eps)

                    loss_c += args.lambda_geo * torch.mean(torch.abs(syn_geo - real_geo))
                    loss_c += args.lambda_feat * torch.mean(torch.abs(syn_feat - real_feat))
                    loss_c += args.lambda_delta * torch.mean(torch.abs(syn_delta - real_delta))
                    loss_c += args.lambda_ratio * torch.mean(torch.abs(syn_ratio - real_ratio))

                loss += (loss_c / max(pair_count, 1)) * args.ppc

            optimizer_img.zero_grad()
            if args.use_rotation:
                optimizer_rot.zero_grad()

            loss.backward()

            optimizer_img.step()
            if args.use_rotation:
                optimizer_rot.step()

            if (it % args.PRINT_EVERY == 0) or (it == args.Iteration):
                loss_avg = loss.item() / num_classes
                print_train_progress(it, args.Iteration, loss_avg)

    print("\n========== Final Result ==========")
    print(f"Best eval iter : {best_eval_iter}")
    print(f"Best eval model: {best_eval_model}")
    print(f"Best mean acc  : {best_eval_acc:.4f}")

    if final_eval_summary is not None:
        for model_name, info in final_eval_summary.items():
            print(
                f"{model_name}: mean={info['mean_acc']:.4f}, std={info['std_acc']:.4f}, "
                f"best_single={info['best_single_acc']:.4f}"
            )


if __name__ == "__main__":
    set_seed(1999)
    main()
