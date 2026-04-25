import torch
import numpy as np
import matplotlib.pyplot as plt


PT_PATH = "./result_h5/best_res_AnchorChain_MODELNET40_H5_PointNet_10ppc.pt"
SAVE_PATH = "syn_pointcloud_5x5.png"

NUM_CLASSES_SHOW = 5
NUM_SAMPLES_PER_CLASS_SHOW = 5
PPC = 10

MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox"
]


def set_axes_equal(ax, x, y, z):
    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def main():
    pkg = torch.load(PT_PATH, map_location="cpu", weights_only=False)

    pointcloud_syn = pkg["pointcloud_syn"]  # [num_classes * ppc, 3, N]
    label_syn = pkg["label_syn"]

    fig = plt.figure(figsize=(15, 15))

    plot_id = 1

    for class_id in range(NUM_CLASSES_SHOW):
        for sample_id in range(NUM_SAMPLES_PER_CLASS_SHOW):
            index = class_id * PPC + sample_id

            pc = pointcloud_syn[index]  # [3, N]
            label = int(label_syn[index])
            class_name = MODELNET40_CLASSES[label]

            pc = pc.detach().cpu().numpy().T  # [N, 3]

            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]

            ax = fig.add_subplot(
                NUM_CLASSES_SHOW,
                NUM_SAMPLES_PER_CLASS_SHOW,
                plot_id,
                projection="3d"
            )

            ax.scatter(x, y, z, s=1)

            ax.set_title(
                f"{class_name}\nclass={label}, idx={index}",
                fontsize=8
            )

            # 第一列额外标注类别名，方便看每一行是什么类
            if sample_id == 0:
                ax.set_ylabel(class_name, fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")

            set_axes_equal(ax, x, y, z)

            plot_id += 1

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    plt.show()

    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()