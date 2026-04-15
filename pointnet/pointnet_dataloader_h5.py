import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


def pc_normalize(pc):
    """
    将点云平移到中心并缩放到单位球，减少尺度与平移差异。
    输入:
        pc: [N, 3]
    返回:
        pc: [N, 3]
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


class H5FolderLoader(Dataset):
    """
    适用于如下目录结构：

    root/
    ├── chair/
    │   ├── train/
    │   │   ├── chair_0001.h5
    │   │   └── ...
    │   └── test/
    │       ├── chair_0101.h5
    │       └── ...
    ├── table/
    │   ├── train/
    │   └── test/
    └── ...

    每个 h5 文件内部默认包含:
        data: [N, 3]
    """

    def __init__(self, root, split='train', normalize=True, dataset_name='data'):
        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.dataset_name = dataset_name

        assert split in ['train', 'test'], "split must be 'train' or 'test'"

        # 扫描所有 h5 文件
        self.data_list = sorted(glob.glob(str(self.root / '*' / split / '*.h5')))
        if len(self.data_list) == 0:
            raise ValueError(f'No h5 files found in: {self.root}/*/{split}/*.h5')

        # 自动建立 类别名 -> 索引
        class_names = sorted({Path(p).parts[-3] for p in self.data_list})
        self.classes = dict(zip(class_names, range(len(class_names))))

        print(f'The size of {split} data is {len(self.data_list)}')
        print('Classes:', self.classes)

    def __len__(self):
        return len(self.data_list)

    def _get_item(self, index):
        h5_path = self.data_list[index]

        # 类别名 = .../class_name/train/xxx.h5 里的 class_name
        class_name = Path(h5_path).parts[-3]
        label = self.classes[class_name]

        with h5py.File(h5_path, 'r') as f:
            point_set = f[self.dataset_name][:]

        point_set = np.asarray(point_set, dtype=np.float32)

        # 兼容 [N,3] / [3,N]
        if point_set.ndim != 2:
            raise ValueError(f'Unexpected shape in {h5_path}: {point_set.shape}')

        if point_set.shape[1] == 3:
            pass
        elif point_set.shape[0] == 3:
            point_set = point_set.T
        else:
            raise ValueError(f'Unexpected shape in {h5_path}: {point_set.shape}')

        # 只保留前三列坐标
        point_set = point_set[:, 0:3]

        if self.normalize:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # 转成 [3, npoints]
        point_set = point_set.T

        return torch.from_numpy(point_set), torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        return self._get_item(index)