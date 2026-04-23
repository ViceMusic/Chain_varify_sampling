import os
import copy
import math
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from utils import *

'''
# 骨架代码

当前蒸馏损失位于 for c in range(num_classes) 内部，
核心是比较 layers_real["x_m"] 与 layers_syn["x_m"] 的均值特征，
并叠加 M3DLoss 对全局特征 x_gf 的约束。

要新增或替换损失，
主要修改这一段：
 real / syn 的特征提取方式、
 loss_c 的定义，
 或删改 m3d_criterion(...) 这一项，而主循环、评估与保存逻辑基本都不用动。
'''

# 基础设置
CONFIG = {
    "method": "DC",
    "dataset": "MODELNET40_H5",
    "model": "PointNet",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # distillation
    "ppc": 10,
    "eval_mode": "S",
    "num_exp": 1,
    "num_eval": 10,
    "epoch_eval_train": 500,
    "Iteration": 1500,
    "eval_interval": 250,

    "lr_img": 10.0,
    "lr_rot": 0.01,
    "lr_net": 0.01,
    "batch_real": 8,
    "batch_train": 8,
    "init": "real",
    "feature_transform": 0,
    "mmdkernel": "gaussian",

    # save / print
    "save_path": "result_h5",
    "mode": "result_h5",
    "PRINT_EVERY": 10,
}
# 一个字典转对象的方法，适配CONFIG，别的不用管
def build_args_from_config(config_dict):
    return SimpleNamespace(**config_dict)
# 随机种子设置这么高感觉有点问题，感觉是做这个的人玩了一手抽卡，然后选了最好的一次写进论文了
def set_seed(seed=1999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
#进度条
def print_train_progress(it, total_it, loss_avg):
    percent = 100.0 * it / max(total_it, 1)
    print(f"[{percent:6.2f}%] Iter {it:04d}/{total_it} | distill_loss = {loss_avg:.4f}")
# 评估当前 synthetic 数据集的性能，返回每个评估模型的结果汇总，以及表现最好的模型和对应的平均准确率。
# 和后面的配合使用，可以把当前最好的 synthetic 数据保存下来，或者在训练过程中监控 synthetic 数据的性能变化。
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
    print("================================")

    # dataset
    npoints, coord_dim, num_classes, dst_train, _, testloader = get_dataset(args, args.dataset)
    args.num_classes = num_classes
    channel = 3

    # fixed evaluation schedule
    eval_it_pool = list(range(0, args.Iteration + 1, args.eval_interval))
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
        # 反正是提供了一些抽取类的方法，主要是按照类别来抽取数据的，get_pointclouds(c, n)就是从类别c中随机抽取n个点云数据。
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
        # 获取同样维度的数据大小，这里是按照类别来组织的，每个类别有ppc个点云数据，每个点云数据是[3, N]的格式。
        pointcloud_tmp = torch.randn(
            size=(num_classes * args.ppc, coord_dim, npoints),
            dtype=torch.float,
            requires_grad=True,
            device=args.device
        )
        pointcloud_tmp = torch.tanh(pointcloud_tmp).detach().clone().requires_grad_(True)

        # 如果初始化方式是 "real"，那么就从真实数据中随机选取一些点云数据来初始化 synthetic 数据。这样做的好处是可以让 synthetic 数据有一个比较好的起点，可能会加速训练过程。
        if args.init == "real":
            for c in range(num_classes):
                pointcloud_tmp.data[c * args.ppc:(c + 1) * args.ppc] = get_pointclouds(c, args.ppc).detach().data

        # 还有标签大小，按照类别来组织，每个类别有ppc个点云数据，所以标签就是每个类别重复ppc次。
        label_syn = torch.tensor(
            [np.ones(args.ppc) * i for i in range(num_classes)],
            dtype=torch.long,
            requires_grad=False,
            device=args.device
        ).view(-1)

        
        optimizer_img = torch.optim.SGD([pointcloud_tmp], lr=args.lr_img, momentum=0.5)
        m3d_criterion = M3DLoss(args.mmdkernel)

        # distillation loop
        for it in range(args.Iteration + 1):

            # 获取数据，当前的 synthetic 数据就是 pointcloud_tmp，标签是 label_syn。
            pointcloud_syn= pointcloud_tmp

            # evaluate current synthetic set
            # 如果合适的话，就会把当前的内容保留下来
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
                    best_data_package = {
                        "pointcloud_syn": copy.deepcopy(pointcloud_syn.detach().cpu()),
                        "label_syn": copy.deepcopy(label_syn.detach().cpu()),
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
            # 每一轮模型都随机初始化一个网络，然后用这个网络来提取 synthetic 数据和真实数据的特征，计算它们之间的差异，作为损失来优化 synthetic 数据和旋转角度。
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
                # 真实数据是随机抽取出ppc个
                # 合成数据是只有ppc
                # 而不是对数据集中所有数据都进行匹配，随机抽取不算是很强的trick，大伙都能接收的一种
                # 提取出两个层的东西
                pc_syn = pointcloud_syn[c * args.ppc:(c + 1) * args.ppc].reshape(
                    args.ppc, coord_dim, npoints
                ).to(args.device)
                pc_real = get_pointclouds(c, args.batch_real)
                with torch.no_grad():
                    _, _, _, _, layers_real = net(pc_real)
                _, _, _, _, layers_syn = net(pc_syn)

                # 对这一整个batch做平均池化，得到每个channel的平均特征值
                # [B,C,N]->[C,N]，也就是每个channel的平均特征值
                # 也就是说只需要改正这里就好了，想要新增或者替换损失的话，主要修改这里的特征提取方式和 loss_c 的定义就好了，其他的评估和保存逻辑基本都不用动。
                real = layers_real["x_m"].mean(dim=0)
                syn = layers_syn["x_m"].mean(dim=0)

                # 计算误差留着下次用
                # 分channel的误差，权重为0.2
                loss_c = (((real - syn) ** 2).sum(dim=0)).mean() * 0.2
                # M3DLoss误差，对全局特征 x_gf 进行约束，权重是0.001
                loss_c += m3d_criterion(layers_real["x_gf"], layers_syn["x_gf"]) * 0.001
                loss += loss_c * args.ppc

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

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