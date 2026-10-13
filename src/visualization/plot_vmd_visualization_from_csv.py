# -*- coding: utf-8 -*-
"""
基于 CSV 源数据重绘 VMD 分解结果与各分量图。

读取 json_to_csv_converter.py 导出的 vmd_signal_data_from_json.csv 与
vmd_imf_stats_from_json.csv，排版配色对齐 plot_vmd_visualization_combined_final.py：
顶部一行全序列总览，下方 5 行 × 3 列分量展示，每种物资用各自的配色方案。
图像输出到 figures/vmd_visualization_from_csv/combined_final/。

用法（项目根目录执行）：
  python src/visualization/plot_vmd_visualization_from_csv.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

warnings.filterwarnings('ignore')

# ---- 项目根路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import configs.config as cfg
    FIGURES_DIR = Path(cfg.FIGURES_DIR)
except Exception:
    FIGURES_DIR = PROJECT_ROOT / "figures"

# ---- 字体配置（与原始文件一致） ----
try:
    FONT_DEJAVUSANS = cfg.FONT_DEJAVUSANS_FILE_PATH
    FONT_SARASAMONOSC_REGULAR = cfg.FONT_SARASAMONOSC_REGULAR_FILE_PATH
    FONT_SARASAMONOSC_BOLD = cfg.FONT_SARASAMONOSC_BOLD_FILE_PATH
    HAS_CUSTOM_FONT = True
except AttributeError:
    HAS_CUSTOM_FONT = False

if HAS_CUSTOM_FONT:
    try:
        fm.fontManager.addfont(FONT_DEJAVUSANS)
        fm.fontManager.addfont(FONT_SARASAMONOSC_REGULAR)
        fm.fontManager.addfont(FONT_SARASAMONOSC_BOLD)
        plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'DejaVu Sans']
    except Exception:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

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

# ---- 配置映射（与原始文件一致） ----
MATERIAL_NAMES = {
    "001": "10kV油浸式变压器",
    "002": "架空绝缘导线",
    "003": "智能电表",
}
MATERIAL_IDS = ["001", "002", "003"]
SUBFIG_TAGS = ["(a)", "(b)", "(c)"]

MATERIAL_COLOR_SCHEMES = {
    "001": {
        "orig": "#2C4058",
        "imfs": ["#C0392B", "#D35400", "#E67E00", "#F39C12"],
    },
    "002": {
        "orig": "#1B8F5A",
        "imfs": ["#16834A", "#22A15F", "#47B881", "#73C6A2"],
    },
    "003": {
        "orig": "#5A2A6E",
        "imfs": ["#1F77B4", "#2E86C1", "#5DADE2", "#85C1E9"],
    }
}

ROW_LABELS = ['原始信号', 'IMF1', 'IMF2', 'IMF3', 'IMF4']
IMF_LABELS_SHORT = ['趋势', '低频', '中频', '高频']

# ---- 微调参数（与原始文件一致） ----
FILL_ALPHA_ORIG = 0.05
FILL_ALPHA_IMF = 0.05
ANNOT_ALPHA = 0.72
ANNOT_LW = 0.34
ROW_LABEL_X_OFFSET = 0.062
BOTTOM_TICK_COUNT = 4

VMD_K = 4

# ---- 特征展示窗口（与原始文件一致） ----
FEATURE_WINDOWS = {
    "001": ("2021-01-01", "2023-12-31"),
    "002": ("2019-01-01", "2024-12-31"),
    "003": ("2022-01-01", "2024-12-31"),
}

# ---- 保存路径 ----
SAVE_DIR = FIGURES_DIR / 'vmd_visualization_from_csv' / 'combined_final'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name, formats=('png', 'pdf')):
    paths = []
    for fmt in formats:
        p = SAVE_DIR / f'{name}.{fmt}'
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    print(f'  [OK] 已保存: {paths[0]}')
    return str(paths[0])


def stylize_axis(ax, show_x=False):
    ax.grid(True, linestyle='--', alpha=0.12, linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=2.8, width=0.55)
    ax.tick_params(axis='both', which='minor', length=1.6, width=0.4)
    if not show_x:
        ax.tick_params(axis='x', labelbottom=False)


def add_row_labels(fig, axes_lower):
    for i, label in enumerate(ROW_LABELS):
        pos = axes_lower[i, 0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        x_left = pos.x0 - ROW_LABEL_X_OFFSET
        fig.text(
            x_left, y_center, label,
            va='center', ha='center', rotation=90,
            fontsize=9.0, fontweight='bold'
        )


def load_data_from_csv():
    """从 CSV 读取 VMD 信号数据和统计特征"""
    signal_path = SAVE_DIR / 'vmd_signal_data_from_json.csv'
    stats_path = SAVE_DIR / 'vmd_imf_stats_from_json.csv'

    if not signal_path.exists() or not stats_path.exists():
        print(f"[ERROR] CSV 文件不存在，请先执行 json_to_csv_converter.py")
        return None, None

    df_signal = pd.read_csv(signal_path, dtype={'material_id': str})
    df_stats = pd.read_csv(stats_path, dtype={'material_id': str})

    # 整理数据格式
    material_data = {}

    for mat_id in MATERIAL_IDS:
        mat_signal = df_signal[df_signal['material_id'] == mat_id].reset_index(drop=True).copy()
        mat_stats = df_stats[df_stats['material_id'] == mat_id].reset_index(drop=True).copy()

        dates = pd.to_datetime(mat_signal['date'])
        demand = mat_signal['demand'].values.astype(float)
        imfs = np.array([mat_signal[f'IMF{k+1}'].values.astype(float) for k in range(VMD_K)])

        # 读取统计特征
        imf_stats = []
        for k in range(VMD_K):
            row = mat_stats.iloc[k]
            imf_stats.append({
                'imf': row['IMF'],
                'label': row['label'],
                'var_pct': row['var_pct'],
                'dominant_period_months': row['dominant_period_months'],
            })

        reconstruction = {
            'R2': float(mat_stats['reconstruction_R2'].iloc[0]),
            'RMSE': float(mat_stats['reconstruction_RMSE'].iloc[0]),
            'MAE': float(mat_stats['reconstruction_MAE'].iloc[0]),
        }

        # 截取特征展示窗口
        start_str, end_str = FEATURE_WINDOWS[mat_id]
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        mask = (dates >= start_dt) & (dates <= end_dt)
        crop_idx = np.where(mask)[0]
        s_idx, e_idx = int(crop_idx[0]), int(crop_idx[-1])

        win_dates = dates.iloc[s_idx:e_idx + 1].reset_index(drop=True)
        win_demand = demand[s_idx:e_idx + 1]
        win_imfs = imfs[:, s_idx:e_idx + 1]

        material_data[mat_id] = {
            'dates': dates,
            'demand': demand,
            'imfs': imfs,
            'imf_stats': imf_stats,
            'reconstruction': reconstruction,
            'win_dates': win_dates,
            'win_demand': win_demand,
            'win_imfs': win_imfs,
            's_idx': s_idx,
            'e_idx': e_idx,
        }

    return material_data, df_stats


def build_unified_vmd_visualization():
    print("=" * 60)
    print(" 基于 CSV 源数据的 VMD 分解总图")
    print("=" * 60)

    # 1. 加载数据
    material_data, df_stats = load_data_from_csv()
    if material_data is None:
        return

    # 2. 创建双 GridSpec 架构画布（与原始文件一致）
    fig = plt.figure(figsize=(15.0, 10.5))

    gs_top = fig.add_gridspec(
        nrows=1, ncols=1,
        left=0.10, right=0.98,
        bottom=0.81, top=0.92
    )
    gs_bottom = fig.add_gridspec(
        nrows=5, ncols=3,
        left=0.10, right=0.98,
        bottom=0.06, top=0.73,
        hspace=0.10, wspace=0.16
    )

    ax_overview = fig.add_subplot(gs_top[0, 0])
    axes_lower = np.empty((5, 3), dtype=object)
    for i in range(5):
        for j in range(3):
            axes_lower[i, j] = fig.add_subplot(gs_bottom[i, j])

    # 3. 绘制顶部总览图
    overview_min_y, overview_max_y = np.inf, -np.inf
    for j, material_id in enumerate(MATERIAL_IDS):
        data = material_data[material_id]
        scheme = MATERIAL_COLOR_SCHEMES[material_id]

        y_min, y_max = np.min(data['demand']), np.max(data['demand'])
        y_scaled = (data['demand'] - y_min) / (y_max - y_min + 1e-12)

        ax_overview.plot(
            data['dates'], y_scaled,
            color=scheme["orig"], linewidth=1.0, alpha=0.85,
            label=f"{material_id} {MATERIAL_NAMES[material_id]} (归一化)"
        )
        overview_min_y = min(overview_min_y, np.min(y_scaled))
        overview_max_y = max(overview_max_y, np.max(y_scaled))

        # 虚线框
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
        ax_overview.text(
            win_start_dt + (win_end_dt - win_start_dt) / 2, 1.05,
            f"展示视窗 [{material_id}]",
            color=scheme["orig"], fontsize=6.8,
            va='bottom', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=scheme["orig"],
                      alpha=ANNOT_ALPHA, lw=ANNOT_LW)
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
        loc='upper left', ncol=3, framealpha=0.88, edgecolor='#cccccc',
        fontsize=7.2, columnspacing=1.0, borderpad=0.3
    )

    # 4. 绘制下方 5x3 截取时段分量图
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

        # Row 0: 原始局部信号
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

        # Row 1~4: IMF1~IMF4
        for k in range(VMD_K):
            ax = axes_lower[k + 1, j]
            imf = win_imfs[k]
            ax.fill_between(x_dates, imf, color=color_imfs[k], alpha=FILL_ALPHA_IMF)
            ax.plot(x_dates, imf, color=color_imfs[k], linewidth=0.88)
            ax.axhline(0, color='#666666', linewidth=0.45, linestyle='-', alpha=0.5)
            stylize_axis(ax, show_x=(k == VMD_K - 1))

            p_val = imf_stats[k]['dominant_period_months']
            p_text = f"{float(p_val):.1f}月" if (p_val is not None and p_val != 'N/A') else "趋势"
            info = f"{IMF_LABELS_SHORT[k]} | Var={imf_stats[k]['var_pct']:.2f}% | T≈{p_text}"
            ax.text(
                0.018, 0.88, info,
                transform=ax.transAxes, fontsize=6.5, va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color_imfs[k], alpha=ANNOT_ALPHA, lw=ANNOT_LW)
            )

        # 底部 X 轴
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

    for j in range(3):
        axes_lower[-1, j].set_xlabel('时间', fontsize=8.2, fontweight='bold')

    add_row_labels(fig, axes_lower)

    fig.suptitle(
        '不同形态电力物资需求序列的全生命周期 VMD 多尺度特征演化对比',
        fontsize=12.0, y=0.993, fontweight='bold'
    )

    # 5. 文件保存
    fig_name = 'fig_vmd_combined_from_csv'
    save_figure(fig, fig_name, formats=('png', 'pdf'))

    print(f"\n  [OK] 图像已保存至: {SAVE_DIR}")
    print(f"      文件名: {fig_name}.png / .pdf")


if __name__ == '__main__':
    build_unified_vmd_visualization()