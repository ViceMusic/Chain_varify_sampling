import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# ============================================================
# 1. 严格对齐 IEEEtran 9pt 配置
# ============================================================
TIMES_PATH = "font/times.ttf"
if not os.path.exists(TIMES_PATH):
    # 如果路径不对，请修改此处
    raise FileNotFoundError(f"未找到字体文件: {TIMES_PATH}")

# 预载字体
# 对于 normalsize 9pt 的论文，图中标题用 9pt 是标准的
times_9pt = fm.FontProperties(fname=TIMES_PATH, size=9)

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
})

# ============================================================
# 2. 物理尺寸计算 (针对 IEEEtran 单栏 A4)
# ============================================================
# IEEEtran A4 栏宽约为 8.89cm = 3.5in
FIG_W = 3.5  
FIG_H = 2.0  # 4行布局的黄金比例

# 布局参数：采用绝对坐标微调，减少 3D 轴自带的空隙
LEFT = 0.02
TOP_LIMIT = 0.90 
AX_W = 0.24  # 增加宽度，让点云显得更大
AX_H = 0.24
COL_GAP = 0.24
ROW_GAP = 0.22
TITLE_Y = 0.94 # 标题位置

PATHS = [
    "./result_h5/res_Selection_MODELNET40_H5_PointNet_10ppc_RandomSelection.pt",
    "./result_h5/res_DC_MODELNET40_H5_PointNet_1ppc_AdaSADM_ppc=1_iter=4000.pt",
    "./result_h5/res_DC_MODELNET40_H5_PointNet_3ppc_AdaSADM_ppc=3_iter=4000.pt",
    "./result_h5/res_DC_MODELNET40_H5_PointNet_5ppc_AdaSADM_ppc=5_iter=4000.pt",
]

COLUMN_TITLES = ["Real", "PPC=1", "PPC=3", "PPC=5"]
CLASS_NAMES_SHOW = ["airplane", "bathtub", "bed", "bench"]
MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
]

# ============================================================
# 3. 绘图核心逻辑
# ============================================================

def set_axes_equal_tight(ax, x, y, z, zoom=0.7):
    """
    zoom 越小，点云在轴内显示的越大。
    这里调小 zoom 使得点云看起来更饱满，从而让字体显得不那么突兀。
    """
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0 * zoom
    mid_x, mid_y, mid_z = (x.max()+x.min())*0.5, (y.max()+y.min())*0.5, (z.max()+z.min())*0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    
    # 列标题
    for col_id in range(len(PATHS)):
        x_center = LEFT + col_id * COL_GAP + AX_W / 2.0
        fig.text(x_center, TITLE_Y, COLUMN_TITLES[col_id], 
                 ha="center", fontproperties=times_9pt)

    for col_id, pt_path in enumerate(PATHS):
        # 加载数据
        pkg = torch.load(pt_path, map_location="cpu", weights_only=False)
        pc_all, lb_all = (pkg["pointcloud_syn"], pkg["label_syn"]) if "pointcloud_syn" in pkg else (pkg["data"][0][0], pkg["data"][0][1])
        
        for row_id, class_name in enumerate(CLASS_NAMES_SHOW):
            cls_id = MODELNET40_CLASSES.index(class_name)
            idx = np.where(lb_all.numpy() == cls_id)[0][0]
            
            pc = pc_all[idx].detach().cpu().numpy()
            if pc.shape[0] == 3: pc = pc.T
            
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
            
            # 创建 3D 子图，并手动控制位置以消除自带留白
            ax_pos = [LEFT + col_id * COL_GAP, 
                      TOP_LIMIT - AX_H - row_id * ROW_GAP, 
                      AX_W, AX_H]
            ax = fig.add_axes(ax_pos, projection="3d")
            
            # 点的大小：在 3.5in 画布下，0.2 比较合适
            cmap = plt.get_cmap(["Blues", "Greens", "Oranges", "Purples"][col_id])
            colors = cmap(0.4 + 0.5 * (z - z.min()) / (z.max() - z.min() + 1e-9))
            
            ax.scatter(x, y, z, s=0.2, c=colors, depthshade=False, edgecolors='none')
            
            ax.view_init(elev=20, azim=-60)
            # 关键：让点云“撑满”子图，视觉上缩小字体占比
            set_axes_equal_tight(ax, x, y, z, zoom=0.68)
            ax.set_axis_off()

    # 保存 PNG (带透明度) 和 SVG
    plt.savefig("syn_result.pdf", dpi=600, pad_inches=0.01)
    plt.savefig("syn_result.png", dpi=600, pad_inches=0.01)
    print("Done! 已生成符合 IEEEtran 9pt 单栏配置的图片。")

if __name__ == "__main__":
    main()