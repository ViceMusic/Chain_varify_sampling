import os
import copy
import random
from types import SimpleNamespace

import numpy as np
import torch

from utils import *


CONFIG = {
    "method": "Selection",
    "selection_method": "FPS",
    "dataset": "MODELNET40_H5",
    "model": "PointNet",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "ppc": 5,
    "eval_mode": "S",
    "num_exp": 1,
    "num_eval": 10,
    "epoch_eval_train": 500,
    "lr_net": 0.01,
    "batch_real": 16,
    "feature_transform": 0,

    "save_path": "result_h5",
    "mode": "result_h5",
    "message": "FPSSelection",
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


def normalize_pointcloud_shape(pointcloud, coord_dim, npoints):
    pointcloud = pointcloud.detach().clone().float()
    if pointcloud.shape == (npoints, coord_dim):
        pointcloud = pointcloud.t().contiguous()
    assert pointcloud.shape == (coord_dim, npoints), f"Unexpected pointcloud shape: {pointcloud.shape}"
    return pointcloud


def pointcloud_descriptor(pointcloud, coord_dim, npoints):
    pointcloud = normalize_pointcloud_shape(pointcloud, coord_dim, npoints)
    mean_xyz = pointcloud.mean(dim=1)
    std_xyz = pointcloud.std(dim=1, unbiased=False)
    min_xyz = pointcloud.min(dim=1)[0]
    max_xyz = pointcloud.max(dim=1)[0]
    return torch.cat([mean_xyz, std_xyz, min_xyz, max_xyz], dim=0)


def build_indices_class(dst_train, num_classes):
    labels_all = [int(dst_train[i][1]) for i in range(len(dst_train))]
    indices_class = [[] for _ in range(num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    return indices_class


def select_fps_in_class(descs, ppc):
    num_samples = descs.shape[0]
    if num_samples == 0:
        raise ValueError("cannot run FPS on an empty class")

    class_mean = descs.mean(dim=0, keepdim=True)
    first = int(torch.cdist(descs, class_mean).view(-1).argmax().item())
    selected_local = [first]
    min_dist = torch.cdist(descs, descs[first:first + 1]).view(-1)
    min_dist[selected_local] = -1.0

    while len(selected_local) < min(ppc, num_samples):
        next_idx = int(min_dist.argmax().item())
        selected_local.append(next_idx)
        new_dist = torch.cdist(descs, descs[next_idx:next_idx + 1]).view(-1)
        min_dist = torch.minimum(min_dist, new_dist)
        min_dist[selected_local] = -1.0

    return selected_local


def select_fps_indices(dst_train, indices_class, ppc, coord_dim, npoints):
    selected_indices_by_class = {}
    for c, indices in enumerate(indices_class):
        if len(indices) == 0:
            raise ValueError(f"class {c} has no training samples")

        descs = torch.stack(
            [pointcloud_descriptor(dst_train[idx][0], coord_dim, npoints) for idx in indices],
            dim=0
        ).float()
        selected_local = select_fps_in_class(descs, ppc)
        selected = [indices[i] for i in selected_local]

        if len(selected) < ppc:
            extra = np.random.choice(indices, size=ppc - len(selected), replace=True).tolist()
            selected += extra

        selected_indices_by_class[c] = [int(idx) for idx in selected]
    return selected_indices_by_class


def build_selected_set(dst_train, selected_indices_by_class, num_classes, ppc, coord_dim, npoints, device):
    pointcloud_list = []
    label_list = []
    for c in range(num_classes):
        for idx in selected_indices_by_class[c]:
            pointcloud_list.append(normalize_pointcloud_shape(dst_train[idx][0], coord_dim, npoints))
            label_list.append(c)

    pointcloud_syn = torch.stack(pointcloud_list, dim=0).to(device)
    label_syn = torch.tensor(label_list, dtype=torch.long, device=device)

    assert pointcloud_syn.shape == (num_classes * ppc, coord_dim, npoints)
    assert label_syn.shape == (num_classes * ppc,)
    return pointcloud_syn, label_syn


def evaluate_selected_set(pointcloud_syn, label_syn, testloader, args, model_eval_pool, num_classes, channel, logger):
    accs_all_exps = {key: [] for key in model_eval_pool}
    accs_all = []

    logger.info("Evaluate selected set")
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
        accs_all_exps[model_eval] += accs

    return accs_all_exps, accs_all


def main():
    args = build_args_from_config(CONFIG)

    os.makedirs(args.mode, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    log_filename = (
        f"{args.mode}/"
        f"{args.method}_{args.dataset}_{args.model}_{args.ppc}ppc_{args.selection_method}Selection"
    )
    logger = build_logger(".", log_filename)
    logger.info("Device: %s", args.device)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Model: %s", args.model)
    logger.info("Selection method: %s", args.selection_method)
    logger.info("ppc: %d", args.ppc)

    npoints, coord_dim, num_classes, dst_train, _, testloader = get_dataset(args, args.dataset)
    args.num_classes = num_classes
    model_eval_pool = get_eval_pool(args.eval_mode, args.model)
    channel = 3

    indices_class = build_indices_class(dst_train, num_classes)
    for c in range(num_classes):
        logger.info("class %d: %d samples", c, len(indices_class[c]))

    data_save = []
    accs_all_exps = {key: [] for key in model_eval_pool}
    accs_all = []
    selected_indices_by_class = {}

    for exp in range(args.num_exp):
        logger.info("\n================== Exp %d ==================\n", exp)
        selected_indices_by_class = select_fps_indices(dst_train, indices_class, args.ppc, coord_dim, npoints)
        pointcloud_syn, label_syn = build_selected_set(
            dst_train, selected_indices_by_class, num_classes, args.ppc, coord_dim, npoints, args.device
        )
        logger.info("pointcloud_syn.shape: %s", tuple(pointcloud_syn.shape))
        logger.info("label_syn.shape: %s", tuple(label_syn.shape))

        exp_accs_all_exps, exp_accs_all = evaluate_selected_set(
            pointcloud_syn, label_syn, testloader, args, model_eval_pool, num_classes, channel, logger
        )
        for key in model_eval_pool:
            accs_all_exps[key] += exp_accs_all_exps[key]
        accs_all += exp_accs_all
        data_save.append([copy.deepcopy(pointcloud_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])

    save_dict = {
        "data": data_save,
        "accs_all_exps": accs_all_exps,
        "selection_method": args.selection_method,
        "ppc": args.ppc,
        "dataset": args.dataset,
        "model": args.model,
        "selected_indices_by_class": selected_indices_by_class,
    }
    save_name = f"res_{args.method}_{args.dataset}_{args.model}_{args.ppc}ppc_{args.selection_method}Selection.pt"
    torch.save(save_dict, os.path.join(args.save_path, save_name))

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
