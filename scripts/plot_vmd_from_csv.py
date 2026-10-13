# -*- coding: utf-8 -*-
"""绘制三种物资的 VMD 分解总图，上方为全周期总览，下方为 5×3 分量网格，
数据来自 VMD_Signal_Source_Data.csv 与 VMD_IMF_Statistics_Summary.csv。"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.signal import periodogram
from pathlib import Path

warnings.filterwarnings('ignore')

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(os.getcwd())
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入配置
try:
    import configs.config as cfg

    FIGURES_DIR = Path(cfg.FIGURES_DIR)
except ImportError:
    FIGURES_DIR = PROJECT_ROOT / "figures"

# ---- 字体配置 ----
try:
    fm.fontManager.addfont(cfg.FONT_SARASAMONOSC_REGULAR_FILE_PATH)
    fm.fontManager.addfont(cfg.FONT_SARASAMONOSC_BOLD_FILE_PATH)
    plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'SimHei', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

# 全局排版参数
plt.rcParams.update({
    'font.size': 8.2,
    'axes.titlesize': 10.0,
    'axes.labelsize': 8.2,
    'xtick.labelsize': 7.1,
    'ytick.labelsize': 7.1,
    'legend.fontsize': 7.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.72,
    'lines.linewidth': 0.95,
    'patch.linewidth': 0.4,
})

# ---- 数据配置 ----
MATERIAL_IDS = ["001", "002", "003"]
MATERIAL_NAMES = {
    "001": "10kV油浸式变压器",
    "002": "架空绝缘导线",
    "003": "智能电表",
}
SUBFIG_TAGS = ["(a)", "(b)", "(c)"]
ROW_LABELS = ['原始信号', 'IMF1', 'IMF2', 'IMF3', 'IMF4']
IMF_LABELS_SHORT = ['趋势', '低频', '中频', '高频']

# 各物资的展示窗口
FEATURE_WINDOWS = {
    "001": ("2021-01-01", "2023-12-31"),
    "002": ("2019-01-01", "2024-12-31"),
    "003": ("2022-01-01", "2024-12-31"),
}

# ---- 配色方案 ----
MATERIAL_COLOR_SCHEMES = {
    "001": {
        "orig": "#2C4058",  # 深蓝灰
        "imfs": ["#C0392B", "#D35400", "#E67E00", "#F39C12"],  # 暖色渐变
    },
    "002": {
        "orig": "#1B8F5A",  # 深松绿
        "imfs": ["#16834A", "#22A15F", "#47B881", "#73C6A2"],  # 渐变绿
    },
    "003": {
        "orig": "#5A2A6E",  # 深郁紫
        "imfs": ["#1F77B4", "#2E86C1", "#5DADE2", "#85C1E9"],  # 渐变蓝
    }
}

# ---- 微调参数 ----
FILL_ALPHA_ORIG = 0.05
FILL_ALPHA_IMF = 0.05
ANNOT_ALPHA = 0.72
ANNOT_LW = 0.34
ROW_LABEL_X_OFFSET = 0.062
BOTTOM_TICK_COUNT = 4


# ---- 数据读取 ----
def load_vmd_data_from_csv(signal_csv, stats_csv):
    """从两个 CSV 读取各物资的 VMD 信号与 IMF 统计信息。"""
    signal_df = pd.read_csv(signal_csv)
    stats_df = pd.read_csv(stats_csv)

    material_data = {}

    for mat_id in MATERIAL_IDS:
        mat_signal = signal_df[signal_df['Material_ID'] == mat_id].reset_index(drop=True)
        mat_stats = stats_df[stats_df['Material_ID'] == mat_id].reset_index(drop=True)

        if mat_signal.empty:
            continue

        # 解析日期
        dates = pd.to_datetime(mat_signal['date'])
        demand = mat_signal['Original_Demand'].values

        # 提取 IMF
        imfs_full = np.vstack([
            mat_signal['IMF1_Trend'].values,
            mat_signal['IMF2_LowFreq'].values,
            mat_signal['IMF3_MidFreq'].values,
            mat_signal['IMF4_HighFreq'].values
        ])

        # 截取窗口数据
        start_str, end_str = FEATURE_WINDOWS[mat_id]
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        mask = (dates >= start_dt) & (dates <= end_dt)
        crop_idx = np.where(mask)[0]
        s_idx, e_idx = int(crop_idx[0]), int(crop_idx[-1])

        win_dates = dates.iloc[s_idx:e_idx + 1].reset_index(drop=True)
        win_demand = demand[s_idx:e_idx + 1]
        win_imfs = imfs_full[:, s_idx:e_idx + 1]

        # 统计信息整理成字典，字段与绘图代码保持一致
        imf_stats = []
        for _, row in mat_stats.iterrows():
            imf_stats.append({
                'imf': row['imf'],
                'label': row['label'],
                'var_pct': row['var_pct'],
                'dominant_period_months': row['dominant_period_months']
            })

        material_data[mat_id] = {
            'dates': dates,
            'demand': demand,
            'imfs_full': imfs_full,
            'imf_stats': imf_stats,
            'win_dates': win_dates,
            'win_demand': win_demand,
            'win_imfs': win_imfs,
            's_idx': s_idx,
            'e_idx': e_idx,
        }

    return material_data


# ---- 绘图工具函数 ----
def stylize_axis(ax, show_x=False):
    ax.grid(True, linestyle='--', alpha=0.12, linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=2.8, width=0.55)
    ax.tick_params(axis='both', which='minor', length=1.6, width=0.4)
    if not show_x:
        ax.tick_params(axis='x', labelbottom=False)


def add_row_labels(fig, axes_lower):
    """在最左侧添加公共行标签"""
    for i, label in enumerate(ROW_LABELS):
        pos = axes_lower[i, 0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        x_left = pos.x0 - ROW_LABEL_X_OFFSET
        fig.text(
            x_left, y_center, label,
            va='center', ha='center', rotation=90,
            fontsize=9.0, fontweight='bold'
        )


# ---- 主绘图函数 ----
def plot_vmd_visualization(signal_csv, stats_csv):
    print("正在生成 VMD 分解总图（基于CSV数据）...")

    # 加载 CSV 数据
    material_data = load_vmd_data_from_csv(signal_csv, stats_csv)

    # 创建上下双 GridSpec 画布
    fig = plt.figure(figsize=(15.0, 10.5))

    gs_top = fig.add_gridspec(
        nrows=1, ncols=1,
        left=0.10, right=0.98, bottom=0.81, top=0.92
    )

    gs_bottom = fig.add_gridspec(
        nrows=5, ncols=3,
        left=0.10, right=0.98, bottom=0.06, top=0.73,
        hspace=0.10, wspace=0.16
    )

    # 顶部总览子图
    ax_overview = fig.add_subplot(gs_top[0, 0])

    # 下方 5×3 分量网格
    axes_lower = np.empty((5, 3), dtype=object)
    for i in range(5):
        for j in range(3):
            axes_lower[i, j] = fig.add_subplot(gs_bottom[i, j])

    # 绘制顶部总览图
    overview_min_y, overview_max_y = np.inf, -np.inf

    for j, material_id in enumerate(MATERIAL_IDS):
        data = material_data[material_id]
        scheme = MATERIAL_COLOR_SCHEMES[material_id]

        # 归一化对比
        y_min, y_max = np.min(data['demand']), np.max(data['demand'])
        y_scaled = (data['demand'] - y_min) / (y_max - y_min + 1e-12)

        ax_overview.plot(
            data['dates'], y_scaled,
            color=scheme["orig"], linewidth=1.0, alpha=0.85,
            label=f"{material_id} {MATERIAL_NAMES[material_id]} (归一化)"
        )

        overview_min_y = min(overview_min_y, np.min(y_scaled))
        overview_max_y = max(overview_max_y, np.max(y_scaled))

        # 绘制特征指示虚线框
        win_start_dt = pd.to_datetime(FEATURE_WINDOWS[material_id][0])
        win_end_dt = pd.to_datetime(FEATURE_WINDOWS[material_id][1])
        x0 = mdates.date2num(win_start_dt)
        x1 = mdates.date2num(win_end_dt)
        width = x1 - x0

        rect = Rectangle(
            (x0, -0.05), width, 1.1,
            fill=False, edgecolor=scheme["orig"],
            linewidth=1.1, linestyle=(0, (4, 2.5)), alpha=0.9
        )
        ax_overview.add_patch(rect)

        # 虚线框标签
        ax_overview.text(
            win_start_dt + (win_end_dt - win_start_dt) / 2, 1.05,
            f"展示视窗 [{material_id}]",
            color=scheme["orig"], fontsize=6.8, va='bottom', ha='center',
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.15', fc='white', ec=scheme["orig"],
                alpha=ANNOT_ALPHA, lw=ANNOT_LW
            )
        )

    ax_overview.set_ylim(-0.1, 1.25)

    all_dates_min = min(material_data[m]['dates'].min() for m in MATERIAL_IDS)
    all_dates_max = max(material_data[m]['dates'].max() for m in MATERIAL_IDS)
    ax_overview.set_xlim(all_dates_min, all_dates_max)

    ax_overview.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_overview.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_overview.xaxis.set_minor_locator(mdates.YearLocator(1))

    ax_overview.grid(True, linestyle='--', alpha=0.12, linewidth=0.5)
    ax_overview.set_ylabel("归一化需求幅值", fontsize=8.2, fontweight='bold')
    ax_overview.set_title(
        "图1. 1995-2025年三种物资全生命周期历史需求演化总览及展示窗口定位",
        fontsize=10.0, fontweight='bold', pad=10
    )

    ax_overview.legend(
        loc='upper left', ncol=3, framealpha=0.88,
        edgecolor='#cccccc', fontsize=7.2,
        columnspacing=1.0, borderpad=0.3
    )

    # 绘制下方各分量子图
    for j, material_id in enumerate(MATERIAL_IDS):
        data = material_data[material_id]
        scheme = MATERIAL_COLOR_SCHEMES[material_id]

        color_orig = scheme["orig"]
        color_imfs = scheme["imfs"]

        win_dates = data['win_dates']
        win_demand = data['win_demand']
        win_imfs = data['win_imfs']
        imf_stats = data['imf_stats']

        x_dates = win_dates.to_numpy()

        # ---- 原始局部信号 ----
        ax = axes_lower[0, j]
        ax.fill_between(x_dates, win_demand, color=color_orig, alpha=FILL_ALPHA_ORIG)
        ax.plot(x_dates, win_demand, color=color_orig, linewidth=1.0)
        stylize_axis(ax, show_x=False)

        ax.set_title(f"{SUBFIG_TAGS[j]} {MATERIAL_NAMES[material_id]}", pad=8, fontweight='bold', fontsize=9.2)

        start_str_box = win_dates.iloc[0].strftime("%Y-%m")
        end_str_box = win_dates.iloc[-1].strftime("%Y-%m")
        ax.text(
            0.018, 0.88, f"窗口: {start_str_box} ~ {end_str_box}",
            transform=ax.transAxes, fontsize=6.8, va='top',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color_orig, alpha=ANNOT_ALPHA, lw=ANNOT_LW)
        )

        # ---- IMF1~4 局部信号 ----
        for k in range(4):
            ax = axes_lower[k + 1, j]
            imf = win_imfs[k]

            ax.fill_between(x_dates, imf, color=color_imfs[k], alpha=FILL_ALPHA_IMF)
            ax.plot(x_dates, imf, color=color_imfs[k], linewidth=0.88)
            ax.axhline(0, color='#666666', linewidth=0.45, linestyle='-', alpha=0.5)
            stylize_axis(ax, show_x=(k == 3))

            p_val = imf_stats[k]['dominant_period_months']
            p_text = f"{p_val:.1f}月" if (p_val is not None and np.isfinite(p_val)) else "趋势"

            info = f"{IMF_LABELS_SHORT[k]} | Var={imf_stats[k]['var_pct']:.2f}% | T≈{p_text}"
            ax.text(
                0.018, 0.88, info,
                transform=ax.transAxes, fontsize=6.5, va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color_imfs[k], alpha=ANNOT_ALPHA, lw=ANNOT_LW)
            )

        # ---- 底部 X 轴刻度 ----
        last_ax = axes_lower[-1, j]
        tick_count = min(BOTTOM_TICK_COUNT, len(win_dates))
        tick_idx = np.linspace(0, len(win_dates) - 1, tick_count, dtype=int)
        tick_dates = [win_dates.iloc[i] for i in tick_idx]
        last_ax.set_xticks(tick_dates)
        last_ax.set_xticklabels(
            [d.strftime('%Y-%m') for d in tick_dates],
            rotation=20, ha='right', fontsize=7.1
        )
        last_ax.tick_params(axis='x', labelbottom=True)
        axes_lower[-1, j].set_xlabel('时间', fontsize=8.2, fontweight='bold')

    # 公共行标签
    add_row_labels(fig, axes_lower)

    # 大标题
    fig.suptitle(
        '不同形态电力物资需求序列的全生命周期 VMD 多尺度特征演化对比（CSV数据驱动）',
        fontsize=12.0, y=0.993, fontweight='bold'
    )

    # 保存
    save_dir = FIGURES_DIR / 'vmd_visualization_from_csv'
    save_dir.mkdir(parents=True, exist_ok=True)

    png_path = save_dir / 'fig_vmd_combined_multi_material_final.png'
    pdf_path = save_dir / 'fig_vmd_combined_multi_material_final.pdf'

    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)

    print(f"✅ 已保存论文插图: {png_path} + PDF")

    # 打印摘要
    print('\n========== 论文终稿总图摘要 ==========')
    for material_id in MATERIAL_IDS:
        if material_id in material_data:
            data = material_data[material_id]
            print(
                f"[{material_id}] {MATERIAL_NAMES[material_id]} | "
                f"全序列长度: {len(data['demand'])} | "
                f"展示视窗: {FEATURE_WINDOWS[material_id][0]} ~ {FEATURE_WINDOWS[material_id][1]}"
            )


if __name__ == '__main__':
    signal_csv = './data/VMD_Signal_Source_Data.csv'
    stats_csv = './data/VMD_IMF_Statistics_Summary.csv'

    if os.path.exists(signal_csv) and os.path.exists(stats_csv):
        plot_vmd_visualization(signal_csv, stats_csv)
    else:
        print(f"❌ 文件未找到: {signal_csv} 或 {stats_csv}")
