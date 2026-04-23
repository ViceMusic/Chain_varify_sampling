import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pointnet import pointnet_dataloader_h5, pointnet_model


DATASET_ROOT = "/root/autodl-tmp/datasets/ModelNet40_h5"

'''
这段代码中并没有蒸馏优化，而是对蒸馏以后的数据集尝试优化
传入的合成数据集格式应该是：[B, 3, N]
传入的标签格式应该是：[B]

可能的调用方式
# 1. 先拿真实测试集
_, _, num_classes, train_dataset, train_loader, test_loader = get_dataset(args)

# 2. 创建一组 synthetic data
syn_images = ...
syn_labels = ...

# 3. 创建一个新模型
net = get_network("PointNet", channel=3, num_classes=num_classes)

# 4. 评估 synthetic set
evaluate_synset(it_eval, net, syn_images, syn_labels, test_loader, args)
'''

# 获取数据集，当前只能使用ModelNet40 H5格式的数据集，返回数据加载器和相关信息。
def get_dataset(args, dataset="MODELNET40_H5", npoints=1024):
    """Only keep the ModelNet40 H5 loading flow used by the current project."""
    if dataset != "MODELNET40_H5":
        raise ValueError(f"Only MODELNET40_H5 is supported, got: {dataset}")

    num_classes = 40
    coord_dim = 3

    train_dataset = pointnet_dataloader_h5.H5FolderLoader(
        root=DATASET_ROOT,
        split="train",
        normalize=True,
        dataset_name="data",
    )
    test_dataset = pointnet_dataloader_h5.H5FolderLoader(
        root=DATASET_ROOT,
        split="test",
        normalize=True,
        dataset_name="data",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_real,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
    )
    return npoints, coord_dim, num_classes, train_dataset, train_loader, test_loader


# synthetic samples 已经被优化成张量了，这个类是包装成一个小训练集，再喂给网络训练。
class TensorDataset(Dataset):
    """Wrap tensors as a dataset for evaluation on synthetic point clouds."""

    def __init__(self, images, labels):
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

# 模型组建，也就是传入信息然后返回对应模型
# 目前是只支持PointNet,在pointnet这个文件夹下面的pointnet_model.py里定义了这个模型的具体结构。
def get_network(model, channel, num_classes, feature_transform=False):
    """Build the PointNet classifier used by the existing training code."""
    if model == "PointNet":
        net = pointnet_model.PointNetCls(k=num_classes, feature_transform=feature_transform)
    else:
        raise ValueError(f"unknown model: {model}")

    gpu_num = torch.cuda.device_count()
    device = "cuda" if gpu_num > 0 else "cpu"
    if gpu_num > 1:
        net = nn.DataParallel(net)
    return net.to(device)

'''
这是一个高斯核函数模块：
先算输入之间的 pairwise 距离
再做 RBF kernel
还用了多带宽核叠加
'''
class RBF(nn.Module):
    """Gaussian RBF kernel used by the M3D matching loss."""

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, l2_distances):
        if self.bandwidth is None:
            n_samples = l2_distances.shape[0]
            return l2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, x):
        l2_distances = torch.cdist(x, x) ** 2
        return torch.exp(
            -l2_distances[None, ...]
            / (self.get_bandwidth(l2_distances) * self.bandwidth_multipliers)[:, None, None]
        ).sum(dim=0)

'''
把两组特征 x 和 y 拼起来
用核函数算相似度矩阵
最后算：
xx
xy
yy
返回 xx - 2xy + yy
'''
class M3DLoss(nn.Module):
    """Feature matching loss used during synthetic point optimization."""

    def __init__(self, kernel_type):
        super().__init__()
        if kernel_type == "gaussian":
            self.kernel = RBF()
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    def forward(self, x, y):
        kernel = self.kernel(torch.vstack([x, y]))
        x_size = x.shape[0]
        xx = kernel[:x_size, :x_size].mean()
        xy = kernel[:x_size, x_size:].mean()
        yy = kernel[x_size:, x_size:].mean()
        return xx - 2 * xy + yy


def get_time():
    """Return a readable timestamp for console and file logging."""
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

# 训练一个epoch，传入模型什么的
def epoch(mode, dataloader, net, optimizer, criterion, args, aug, calc_classwise_acc=False):
    """Run one train or test epoch on point-cloud batches."""
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if calc_classwise_acc:
        num_classes = args.num_classes
        correct_per_class = [0] * num_classes
        total_per_class = [0] * num_classes

    predictions_per_sample = []
    net.train(mode == "train")
    if mode != "train":
        net.eval()

    for datum in dataloader:
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output, feats, _, _, _ = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if calc_classwise_acc:
            _, predicted = torch.max(output, 1)
            for label, prediction in zip(lab, predicted):
                if label == prediction:
                    correct_per_class[label] += 1
                total_per_class[label] += 1
                predictions_per_sample.append((label.item(), prediction.item()))

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    acc_test_per_class = None
    if calc_classwise_acc:
        acc_test_per_class = [0.0] * num_classes
        for class_idx in range(num_classes):
            if total_per_class[class_idx] > 0:
                acc_test_per_class[class_idx] = correct_per_class[class_idx] / total_per_class[class_idx]

    return loss_avg, acc_avg, acc_test_per_class, predictions_per_sample

# 点云归一化函数
def pc_normalize(pc):
    """Normalize a single point cloud represented as a numpy array."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
def pc_normalize_batch(pc):
    """Normalize a batch of point clouds represented as torch tensors."""
    centroid = torch.mean(pc, dim=2, keepdims=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1, keepdims=True)), dim=2, keepdims=True).values
    pc = pc / m
    return pc

# 随机种子设置
def seed(seed=42):
    """Set deterministic seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
def seed_worker(_worker_id):
    """Set worker-local random seeds for dataloader reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 验证核心代码，验证合成数据集能不能用
'''
一个新模型 net
一组合成训练数据 images_train
合成标签 labels_train
真实测试集 testloader
'''
def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    """Train a fresh model on synthetic data and evaluate it on the H5 test set."""
    # 准备数据
    net = net.to(args.device)# 移动到GPU
    images_train = pc_normalize_batch(images_train).to(args.device) # 点云归一化，然后移动到GPU
    labels_train = labels_train.to(args.device)                     # 标签也移动到GPU
    # 读取超参数
    lr = float(args.lr_net)
    epoch_count = int(args.epoch_eval_train)
    lr_schedule = [epoch_count // 2 + 1]
    
    # 设置模型训练的优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # 把合成以后的训练集也包装成一个小数据集，喂给网络训练。
    dst_train = TensorDataset(images_train, labels_train)
    generator = torch.Generator()
    generator.manual_seed(0)

    # 划分批次之类的
    trainloader = torch.utils.data.DataLoader(
        dst_train,
        batch_size=args.batch_real,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # 开始训练模型
    start = time.time()
    best_acc = -1
    best_per_class = [0.0] * args.num_classes
    best_prediction = []

    for ep in range(epoch_count + 1):
        loss_train, acc_train, _, _ = epoch("train", trainloader, net, optimizer, criterion, args, aug=False)

        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        if ep % 10 == 0 or ep == epoch_count:
            loss_test, acc_test, acc_test_per_class, predictions_per_sample = epoch(
                "test", testloader, net, optimizer, criterion, args, aug=False, calc_classwise_acc=True
            )
            if acc_test > best_acc:
                best_acc = acc_test
                best_per_class = acc_test_per_class
                best_prediction = predictions_per_sample

    time_train = time.time() - start
    # 合成数据集训练结果搞定

    print(
        "%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f"
        % (get_time(), it_eval, epoch_count, int(time_train), loss_train, acc_train, best_acc)
    )
    return net, acc_train, best_acc, best_per_class, best_prediction

# 模型评估返回池，不是核心代码
def get_eval_pool(eval_mode, model):
    """Return the evaluation model pool used by the training script."""
    if eval_mode == "S":
        return [model]
    if eval_mode == "SSS":
        return [model, "PointNetPlusPlus"]
    raise ValueError(f"Unsupported eval_mode: {eval_mode}")

# 工程打印代码
def build_logger(work_dir, cfgname):
    """Create a file and console logger for training progress."""
    assert cfgname is not None
    log_file = cfgname + ".log"
    log_path = os.path.join(work_dir, log_file)

    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    logger.propagate = False
    return logger
