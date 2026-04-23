import logging
import math
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pointnet.pointnet_model import PointNetCls


CONFIG = {
    "dataset_root": "/root/autodl-tmp/datasets/ModelNet40_h5",
    "local_test_root": "test_h5",
    "prefer_local_test": False,
    "dataset_name": "data",
    "num_classes": 40,
    "input_points": 1024,
    "sample_points": 256,
    "kmedoids_iters": 5,
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "feature_transform": False,
    "seed": 1999,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_file": "k_medoids_method_log.txt",
}


MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
    "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
    "mantel", "monitor", "night_stand", "person", "piano", "plant",
    "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
    "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]


def set_seed(seed):
    """Keep the training and sampling steps reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_logger(log_path):
    """Write training progress to a txt file in the current methods folder."""
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def pc_normalize(pc):
    """Normalize a point cloud into a unit sphere around the centroid."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if scale > 0:
        pc = pc / scale
    return pc


def resolve_dataset_root(config):
    """Prefer the full server path, and fall back to local test data when asked."""
    server_root = Path(config["dataset_root"])
    local_root = ROOT_DIR / config["local_test_root"]
    if config["prefer_local_test"] and local_root.exists():
        return local_root
    if server_root.exists():
        return server_root
    return local_root


def split_flat_files(files, split):
    """Create a minimal train/test split when only flat local test files exist."""
    if len(files) <= 1:
        return files
    pivot = max(1, math.ceil(len(files) * 0.8))
    if split == "train":
        return files[:pivot]
    return files[pivot:] if pivot < len(files) else files[-1:]


def infer_label_from_path(path):
    """Infer the ModelNet40 class id from either the parent folder or the filename."""
    class_name = path.parent.parent.name
    if class_name in MODELNET40_CLASSES:
        return MODELNET40_CLASSES.index(class_name)

    stem_prefix = path.stem.split("_")[0]
    if stem_prefix in MODELNET40_CLASSES:
        return MODELNET40_CLASSES.index(stem_prefix)

    raise ValueError(f"Unable to infer label from path: {path}")


def load_h5_points(path, dataset_name):
    """Load a single H5 point cloud and return it as a float32 numpy array."""
    with h5py.File(path, "r") as handle:
        points = handle[dataset_name][:]
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Unexpected point shape in {path}: {points.shape}")
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T
    return points[:, :3]


def k_medoids_sample(points, num_samples, max_iters):
    """Select representative points with a simple k-medoids refinement loop."""
    num_points = points.shape[0]
    if num_points <= num_samples:
        return points

    step = max(1, num_points // num_samples)
    medoid_indices = torch.arange(0, step * num_samples, step, device=points.device)[:num_samples]
    medoid_indices = medoid_indices.clamp(max=num_points - 1)

    for _ in range(max_iters):
        medoids = points[medoid_indices]
        distances = torch.cdist(points, medoids)
        assignments = distances.argmin(dim=1)
        new_indices = medoid_indices.clone()

        for cluster_id in range(num_samples):
            member_indices = torch.where(assignments == cluster_id)[0]
            if member_indices.numel() == 0:
                continue
            members = points[member_indices]
            intra_dist = torch.cdist(members, members)
            best_member = intra_dist.sum(dim=1).argmin()
            new_indices[cluster_id] = member_indices[best_member]

        if torch.equal(new_indices, medoid_indices):
            break
        medoid_indices = new_indices

    return points[medoid_indices]


def ensure_num_points(points, target_points):
    """Pad or trim the sampled points so the classifier sees a fixed size."""
    num_points = points.shape[0]
    if num_points == target_points:
        return points
    if num_points > target_points:
        return points[:target_points]
    repeat_indices = torch.arange(target_points - num_points, device=points.device) % max(num_points, 1)
    return torch.cat([points, points[repeat_indices]], dim=0)


class ModelNet40H5Dataset(Dataset):
    """Load either the full ModelNet40 H5 tree or the flat local test directory."""

    def __init__(self, root, split, dataset_name, input_points, sample_points, kmedoids_iters):
        self.root = Path(root)
        self.split = split
        self.dataset_name = dataset_name
        self.input_points = input_points
        self.sample_points = sample_points
        self.kmedoids_iters = kmedoids_iters

        hierarchical_files = sorted(self.root.glob(f"*/{split}/*.h5"))
        if hierarchical_files:
            self.files = hierarchical_files
            self.is_hierarchical = True
        else:
            flat_files = sorted(self.root.glob("*.h5"))
            self.files = split_flat_files(flat_files, split)
            self.is_hierarchical = False

        if not self.files:
            raise ValueError(f"No H5 files found for split={split} under {self.root}")

    def __len__(self):
        """Return the number of samples in the current split."""
        return len(self.files)

    def __getitem__(self, index):
        """Read one point cloud, run k-medoids sampling, and return tensor data."""
        file_path = self.files[index]
        points = load_h5_points(file_path, self.dataset_name)
        points = points[:self.input_points]
        points = pc_normalize(points)

        point_tensor = torch.from_numpy(points)
        sampled = k_medoids_sample(point_tensor, self.sample_points, self.kmedoids_iters)
        sampled = ensure_num_points(sampled, self.sample_points)
        sampled = sampled.T.contiguous()

        label = infer_label_from_path(file_path)
        return sampled, torch.tensor(label, dtype=torch.long)


def build_dataloaders(config):
    """Create train and test loaders that match the existing project workflow."""
    dataset_root = resolve_dataset_root(config)
    train_dataset = ModelNet40H5Dataset(
        root=dataset_root,
        split="train",
        dataset_name=config["dataset_name"],
        input_points=config["input_points"],
        sample_points=config["sample_points"],
        kmedoids_iters=config["kmedoids_iters"],
    )
    test_dataset = ModelNet40H5Dataset(
        root=dataset_root,
        split="test",
        dataset_name=config["dataset_name"],
        input_points=config["input_points"],
        sample_points=config["sample_points"],
        kmedoids_iters=config["kmedoids_iters"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    return train_loader, test_loader, dataset_root


def run_epoch(model, loader, optimizer, criterion, device, train_mode):
    """Run one training or evaluation epoch and report average loss and accuracy."""
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for points, labels in loader:
        points = points.float().to(device)
        labels = labels.to(device)

        logits, _, _, _, _ = model(points)
        loss = criterion(logits, labels)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / max(total_samples, 1)
    average_acc = total_correct / max(total_samples, 1)
    return average_loss, average_acc


def main():
    """Train and evaluate a simple ModelNet40 classifier on k-medoids points."""
    set_seed(CONFIG["seed"])
    log_path = Path(__file__).resolve().parent / CONFIG["log_file"]
    logger = build_logger(log_path)

    train_loader, test_loader, dataset_root = build_dataloaders(CONFIG)
    logger.info("Dataset root: %s", dataset_root)
    logger.info("Train samples: %d | Test samples: %d", len(train_loader.dataset), len(test_loader.dataset))

    model = PointNetCls(
        k=CONFIG["num_classes"],
        feature_transform=CONFIG["feature_transform"],
    ).to(CONFIG["device"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    criterion = nn.NLLLoss()

    best_test_acc = 0.0
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, CONFIG["device"], train_mode=True
        )
        test_loss, test_acc = run_epoch(
            model, test_loader, optimizer, criterion, CONFIG["device"], train_mode=False
        )
        best_test_acc = max(best_test_acc, test_acc)
        logger.info(
            "Epoch %03d | train_loss=%.4f train_acc=%.4f | test_loss=%.4f test_acc=%.4f | best=%.4f",
            epoch,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
            best_test_acc,
        )

    logger.info("Finished at %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == "__main__":
    main()
