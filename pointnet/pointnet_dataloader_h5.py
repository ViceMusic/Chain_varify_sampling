import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# 进行归一化，将点云数据调整到单位球内，中心在原点。
# 也就是说这里做的是：把中心放到 (0,0,0)，然后把所有点的距离缩放到不超过1。
def pc_normalize(pc):
    """Normalize [N, 3] point coordinates into a unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if scale > 0:
        pc = pc / scale
    return pc

# 然后把数据转换成PointNet需要的输入格式，即[3, N]的float tensor，以及对应的long标签tensor。
class H5FolderLoader(Dataset):
    """Read ModelNet40 H5 files stored as root/class_name/{train,test}/*.h5."""

    def __init__(self, root, split="train", normalize=True, dataset_name="data"):
        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.dataset_name = dataset_name

        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")

        self.data_list = sorted(glob.glob(str(self.root / "*" / split / "*.h5")))
        if not self.data_list:
            raise ValueError(f"No h5 files found in: {self.root}/*/{split}/*.h5")

        class_names = sorted({Path(path).parts[-3] for path in self.data_list})
        self.classes = dict(zip(class_names, range(len(class_names))))

        print(f"The size of {split} data is {len(self.data_list)}")
        print("Classes:", self.classes)

    def __len__(self):
        """Return the number of H5 files for the current split."""
        return len(self.data_list)

    def _get_item(self, index):
        """Load one H5 file, normalize it, and convert it to PointNet input format."""
        h5_path = self.data_list[index]
        class_name = Path(h5_path).parts[-3]
        label = self.classes[class_name]

        with h5py.File(h5_path, "r") as handle:
            point_set = handle[self.dataset_name][:]

        point_set = np.asarray(point_set, dtype=np.float32)
        if point_set.ndim != 2 or point_set.shape[1] != 3:
            raise ValueError(f"Expected [N, 3] data in {h5_path}, got {point_set.shape}")

        if self.normalize:
            point_set = pc_normalize(point_set)

        point_set = point_set.T
        return torch.from_numpy(point_set), torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        """Return one sample as ([3, N] float tensor, long label tensor)."""
        return self._get_item(index)
