import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm


# ============================================================
# IEEEtran conference 9pt single-column figure style
# 对应：
# \documentclass[conference,9pt,a4paper]{IEEEtran}
#
# LaTeX 中必须这样插入：
# \includegraphics[width=\columnwidth]{ratio5.pdf}
#
# 不要写 height。
# 不要写 width=\textwidth，除非你用的是 figure* 跨双栏图。
#
# 本代码目标：
# 1. 图片物理宽度对应 IEEE 双栏论文中的单栏宽度
# 2. 所有文字使用本地 Times 字体
# 3. 所有文字严格 9pt
# 4. 不使用 bbox_inches="tight"
# 5. 不使用 tight_layout
# ============================================================


# ============================================================
# 1. Local Times font
# ============================================================

TIMES_PATH = "font/times.ttf"

if not os.path.exists(TIMES_PATH):
    raise FileNotFoundError(f"未找到字体文件: {TIMES_PATH}")

times_9pt = fm.FontProperties(fname=TIMES_PATH, size=9)


mpl.rcParams.update({
    # 不依赖系统字体 fallback
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # 数学字体接近 Times
    "mathtext.fontset": "stix",

    # 线条参数
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})


# ============================================================
# 2. Data
# ============================================================

iters = np.array([
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
    1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500,
    2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300,
    3400, 3500, 3600, 3700, 3800, 3900
])

x_m_ratio = np.array([
    0.970, 0.850, 0.590, 0.410, 0.300, 0.240, 0.175, 0.115, 0.070, 0.072,
    0.055, 0.040, 0.032, 0.028, 0.025, 0.020, 0.016, 0.012,
    0.015, 0.011, 0.008, 0.012, 0.006, 0.008, 0.006, 0.006,
    0.004, 0.004, 0.003, 0.003, 0.0025, 0.0020, 0.0020, 0.0015,
    0.0015, 0.0012, 0.0010, 0.0008, 0.0006, 0.0005
])

x_2_ratio = np.array([
    0.025, 0.130, 0.350, 0.495, 0.545, 0.535, 0.415, 0.320, 0.200, 0.192,
    0.150, 0.110, 0.085, 0.075, 0.058, 0.048, 0.035, 0.030,
    0.032, 0.026, 0.020, 0.030, 0.014, 0.018, 0.014, 0.014,
    0.010, 0.009, 0.007, 0.008, 0.007, 0.0045, 0.0045, 0.0040,
    0.0038, 0.0035, 0.0032, 0.0028, 0.0025, 0.0022
])

x_1_ratio = 1.0 - x_m_ratio - x_2_ratio
x_1_ratio = np.clip(x_1_ratio, 0.0, 1.0)


# ============================================================
# 3. Figure size
# ============================================================
# IEEE 双栏论文中的单栏宽度通常约为 3.45--3.50 inch。
#
# 如果你的 LaTeX 中是：
# \includegraphics[width=\columnwidth]{ratio5.pdf}
#
# 那这里就应该使用 3.45 或 3.50。
#
# 高度可以自己调；高度不会影响字体实际 9pt，
# 只影响图中内容是否显得拥挤。
# ============================================================

FIG_W = 3.50
FIG_H = 2.55

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)


# ============================================================
# 4. Plot
# ============================================================

markersize = 1.8
linewidth = 0.75
markeredgewidth = 0.35


ax.plot(
    iters,
    x_m_ratio,
    marker="o",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    linewidth=linewidth,
    label=r"$x_m$ loss ratio"
)

ax.plot(
    iters,
    x_2_ratio,
    marker="o",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    linewidth=linewidth,
    label=r"$x_2$ loss ratio"
)

ax.plot(
    iters,
    x_1_ratio,
    marker="o",
    markersize=markersize,
    markeredgewidth=markeredgewidth,
    linewidth=linewidth,
    label=r"$x_1$ loss ratio"
)


# ============================================================
# 5. Text: all local Times 9pt
# ============================================================

ax.set_title(
    "Layer-wise Loss Contribution during Our method",
    fontproperties=times_9pt,
    fontweight="normal",
    pad=4
)

ax.set_xlabel(
    "Iteration",
    fontproperties=times_9pt,
    labelpad=2
)

ax.set_ylabel(
    "Loss Contribution Ratio",
    fontproperties=times_9pt,
    labelpad=2
)


# ============================================================
# 6. Axes
# ============================================================

ax.set_xlim(-100, 4000)
ax.set_ylim(-0.04, 1.04)

ax.set_xticks(np.arange(0, 4001, 1000))
ax.set_yticks(np.arange(0.0, 1.01, 0.2))

ax.tick_params(
    axis="both",
    which="major",
    length=3,
    width=0.7,
    pad=2
)

# tick labels also use local Times 9pt
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(times_9pt)

for spine in ax.spines.values():
    spine.set_linewidth(0.7)


# ============================================================
# 7. Legend: local Times 9pt
# ============================================================

legend = ax.legend(
    loc="center right",
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    borderpad=0.25,
    handlelength=1.30,
    handletextpad=0.40,
    labelspacing=0.20,
    borderaxespad=0.35,
    prop=times_9pt
)

legend.get_frame().set_linewidth(0.6)


# ============================================================
# 8. Manual layout
# ============================================================
# 不用 tight_layout。
# 不用 bbox_inches="tight"。
#
# 这几个数控制子图区域：
# left   越小，坐标区越靠左，绘图区越宽
# right  越大，绘图区越宽
# bottom 越小，绘图区越高，但可能压到 xlabel
# top    越大，绘图区越高，但可能压到 title
#
# 当前这组是为了：
# 1. 保留 9pt title
# 2. 保留 9pt xlabel / ylabel
# 3. 保留 9pt ticks
# 4. 尽量放大坐标轴绘图区
# ============================================================

fig.subplots_adjust(
    left=0.120,
    right=0.960,
    bottom=0.185,
    top=0.885
)


# ============================================================
# 9. Save
# ============================================================
# 不要加 bbox_inches="tight"。
# 否则 PDF 的物理边界会被裁剪，LaTeX 再 width=\columnwidth 时会缩放字体。
# ============================================================

plt.savefig("ratio5.pdf", dpi=300, pad_inches=0.01, facecolor="white")
plt.savefig("ratio5.png", dpi=300, pad_inches=0.01, facecolor="white")
plt.savefig("ratio5.svg", dpi=300, pad_inches=0.01, facecolor="white")

plt.show()