# -*- coding: utf-8 -*-
"""
三种物资 VMD 分解统一总图（全序列计算 + 局部视窗截取）。

依次处理 001/002/003 三种物资，每种都走项目标准 preprocess 管道读数据，并在
全序列上做 VMD(K=4, alpha=2000)，保证分解口径与主模型一致、不引入局部截取的
边界效应。下方分量图只截取特征时段展示，顶部再加一行全序列总览，用虚线框标出
截取的区间。

排版为 5 行 × 3 列（第 1 行原始信号，第 2~5 行 IMF1~IMF4），公共行标签外移、
信息框减淡、填充低饱和、时间刻度精简。导出 PNG/PDF 并保存对应的数值矩阵（JSON/Excel）。

用法：
  python plot_vmd_visualization_combined_final.py
"""

import os
import sys
import json
import importlib
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.signal import periodogram

warnings.filterwarnings('ignore')

# ---- 项目根路径 ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# ---- 物资映射 ----
MATERIAL_NAMES = {
    "001": "10kV油浸式变压器",
    "002": "架空绝缘导线",
    "003": "智能电表",
}
MATERIAL_IDS = ["001", "002", "003"]
SUBFIG_TAGS = ["(a)", "(b)", "(c)"]

# ---- 配置导入 ----
import configs.config as cfg

importlib.reload(cfg)

SIM_DATASET_PATH = cfg.SIM_DATASET_PATH
DATA_FIGURES_DIR = cfg.DATA_FIGURES_DIR
FIGURES_DIR = cfg.FIGURES_DIR
TRAIN_RATIO = cfg.TRAIN_RATIO
VAL_RATIO = cfg.VAL_RATIO
DEMAND_COL = cfg.DEMAND_COL

try:
    FONT_DEJAVUSANS = cfg.FONT_DEJAVUSANS_FILE_PATH
    FONT_SARASAMONOSC_REGULAR = cfg.FONT_SARASAMONOSC_REGULAR_FILE_PATH
    FONT_SARASAMONOSC_BOLD = cfg.FONT_SARASAMONOSC_BOLD_FILE_PATH
    HAS_CUSTOM_FONT = True
except AttributeError:
    HAS_CUSTOM_FONT = False

from src.data.preprocess import (
    load_raw_data,
    clean_data_train,
    clean_data_with_stats,
    split_by_time,
)

# ---- 字体配置 ----
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

# ---- 全局排版配置 ----
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

# ---- 保存路径 ----
SAVE_DIR = os.path.join(FIGURES_DIR, 'vmd_visualization', 'combined_final')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_FIGURES_DIR, exist_ok=True)

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

ROW_LABELS = ['原始信号', 'IMF1', 'IMF2', 'IMF3', 'IMF4']
IMF_LABELS_SHORT = ['趋势', '低频', '中频', '高频']

# ---- 微调参数 ----
FILL_ALPHA_ORIG = 0.05
FILL_ALPHA_IMF = 0.05

ANNOT_ALPHA = 0.72
ANNOT_LW = 0.34

# 决定公共行标签距离最左侧子图 Y 轴的水平距离
ROW_LABEL_X_OFFSET = 0.062
BOTTOM_TICK_COUNT = 4

# ---- VMD 参数 ----
VMD_K = 4
VMD_ALPHA = 2000
VMD_TOL = 1e-7

try:
    from vmdpy import VMD
except ImportError:
    VMD = None

# ---- 特征展示窗口 ----
FEATURE_WINDOWS = {
    "001": ("2021-01-01", "2023-12-31"),
    "002": ("2019-01-01", "2024-12-31"),
    "003": ("2022-01-01", "2024-12-31"),
}


# ---- 通用工具 ----
def save_figure(fig, name, formats=('png', 'pdf')):
    paths = []
    for fmt in formats:
        p = os.path.join(SAVE_DIR, f'{name}.{fmt}')
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    suffix = ' + PDF' if len(paths) > 1 else ''
    print(f'✅ 已保存论文插图: {paths[0]}{suffix}')
    return paths[0]


def save_figure_data(fig_name, data_dict):
    json_path = os.path.join(SAVE_DIR, f'{fig_name}_data.json')
    xlsx_path = os.path.join(SAVE_DIR, f'{fig_name}_data.xlsx')

    json_safe = {}
    excel_sheets = {}

    for key, val in data_dict.items():
        if isinstance(val, pd.DataFrame):
            json_safe[key] = val.to_dict(orient='records')
            excel_sheets[key[:31]] = val
        elif isinstance(val, dict) or isinstance(val, list):
            json_safe[key] = val
        else:
            json_safe[key] = val

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe, f, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet_name, df in excel_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        meta_rows = []
        for key, val in json_safe.items():
            if isinstance(val, (dict, list)):
                meta_rows.append({"key": key, "value": json.dumps(val, ensure_ascii=False)})
            else:
                meta_rows.append({"key": key, "value": val})
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name='meta', index=False)

    print(f'✅ 相关计算数据矩阵已导出: {json_path} + {xlsx_path}')


def safe_float(v):
    if isinstance(v, np.ndarray):
        v = np.squeeze(v)
        if v.ndim == 0:
            return float(v)
        return float(v.mean())
    return float(v)


# ---- 数据管道加载 ----
def load_full_demand_signal_for_material(material_id):
    """重载配置后，走项目标准 preprocess 逻辑读出一条完整需求序列"""
    os.environ["MATERIAL_SCENARIO"] = material_id
    import configs.config as cfg_local
    importlib.reload(cfg_local)

    sim_dataset_path = cfg_local.SIM_DATASET_PATH
    train_ratio = cfg_local.TRAIN_RATIO
    val_ratio = cfg_local.VAL_RATIO
    demand_col = cfg_local.DEMAND_COL

    df = load_raw_data(sim_dataset_path)

    train_raw, val_raw, test_raw = split_by_time(df, train_ratio, val_ratio)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    full_clean = pd.concat([train_clean, val_clean, test_clean], axis=0).reset_index(drop=True)
    demand = full_clean[demand_col].values.astype(float)

    date_col = None
    for candidate in ['date', 'Date', 'ds', 'month', 'Month', '年月', '日期', 'time', 'Time']:
        if candidate in full_clean.columns:
            date_col = candidate
            break

    if date_col is not None:
        dates = pd.to_datetime(full_clean[date_col])
    else:
        dates = pd.date_range(start='2015-01-01', periods=len(full_clean), freq='MS')

    return full_clean, pd.Series(dates), demand, demand_col


# ---- VMD 分解与统计 ----
def vmd_decompose(signal, K=4, alpha=2000, tol=1e-7):
    if VMD is None:
        raise ImportError("未检测到安装 vmdpy，请先在您的终端执行: pip install vmdpy")

    sig_mean = np.mean(signal)
    signal_centered = signal - sig_mean

    u, _, _ = VMD(signal_centered, alpha, 0, K, 0, 1, tol)

    n_in = len(signal)
    n_out = u.shape[1]

    if n_out < n_in:
        pad = np.zeros((u.shape[0], n_in - n_out))
        for k in range(u.shape[0]):
            if n_out >= 2:
                slope = u[k, -1] - u[k, -2]
                pad[k] = u[k, -1] + slope * np.arange(1, n_in - n_out + 1)
            else:
                pad[k] = u[k, -1]
        u = np.hstack([u, pad])
    elif n_out > n_in:
        u = u[:, :n_in]

    # 将均值加回分量 1 (代表长周期趋势项)，使加性重构等于原信号
    u[0, :] = u[0, :] + sig_mean
    return u


def calc_imf_stats(signal, imfs):
    """在完整序列上统计各 IMF 的方差贡献、主周期，以及重构精度"""
    total_var = np.var(signal) + 1e-12
    imf_stats = []

    for i, imf in enumerate(imfs):
        freq, psd = periodogram(imf, fs=1)
        if len(psd) > 1:
            peak_idx = np.argmax(psd[1:]) + 1
            dominant_freq = safe_float(freq[peak_idx])
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.nan
        else:
            dominant_freq = np.nan
            dominant_period = np.nan

        var_pct = np.var(imf) / total_var * 100

        imf_stats.append({
            'imf': f'IMF{i + 1}',
            'label': IMF_LABELS_SHORT[i],
            'mean': round(float(np.mean(imf)), 4),
            'std': round(float(np.std(imf)), 4),
            'var_pct': round(float(var_pct), 2),
            'dominant_freq': round(float(dominant_freq), 6) if not np.isnan(dominant_freq) else None,
            'dominant_period_months': round(float(dominant_period), 1) if not np.isnan(dominant_period) else None,
        })

    recon = imfs.sum(axis=0)
    residual = signal - recon

    reconstruction = {
        'RMSE': float(np.sqrt(np.mean(residual ** 2))),
        'MAE': float(np.mean(np.abs(residual))),
        'R2': float(1 - np.var(residual) / (np.var(signal) + 1e-12)),
    }

    return imf_stats, reconstruction


# ---- 轴样式统一化 ----
def stylize_axis(ax, show_x=False):
    ax.grid(True, linestyle='--', alpha=0.12, linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=2.8, width=0.55)
    ax.tick_params(axis='both', which='minor', length=1.6, width=0.4)
    if not show_x:
        ax.tick_params(axis='x', labelbottom=False)


# ---- 总图排版构建 ----
def add_row_labels(fig, axes_lower):
    """在最左侧添加公共行标签，向外偏移避开 Y 轴刻度"""
    for i, label in enumerate(ROW_LABELS):
        pos = axes_lower[i, 0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        # pos.x0 是左边界，向左外推 ROW_LABEL_X_OFFSET
        x_left = pos.x0 - ROW_LABEL_X_OFFSET
        fig.text(
            x_left, y_center, label,
            va='center', ha='center',
            rotation=90,
            fontsize=9.0,
            fontweight='bold'
        )


def build_unified_vmd_visualization():
    """核心制图函数：双 GridSpec 布局，顶部总览行与下方分解网格分开，避免重叠"""
    # ---- A. 数据准备与计算 ----
    material_data = {}

    for material_id in MATERIAL_IDS:
        df, dates, demand, demand_col = load_full_demand_signal_for_material(material_id)

        # 1. 对全序列数据运行 VMD
        imfs_full = vmd_decompose(demand, K=VMD_K, alpha=VMD_ALPHA, tol=VMD_TOL)
        imf_stats, reconstruction = calc_imf_stats(demand, imfs_full)

        # 2. 定位特征展示视窗
        start_str, end_str = FEATURE_WINDOWS[material_id]
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)

        mask = (dates >= start_dt) & (dates <= end_dt)
        crop_idx = np.where(mask)[0]
        s_idx, e_idx = int(crop_idx[0]), int(crop_idx[-1])

        win_dates = dates.iloc[s_idx:e_idx + 1].reset_index(drop=True)
        win_demand = demand[s_idx:e_idx + 1]
        win_imfs = imfs_full[:, s_idx:e_idx + 1]

        # 保存导出数据结构
        signal_table = pd.DataFrame({
                        'date': dates.astype(str),
            'demand': demand,
            **{f'IMF{i+1}': imfs_full[i] for i in range(VMD_K)}
        })

        material_data[material_id] = {
            'dates': dates,
            'demand': demand,
            'imfs_full': imfs_full,
            'imf_stats': imf_stats,
            'reconstruction': reconstruction,
            'win_dates': win_dates,
            'win_demand': win_demand,
            'win_imfs': win_imfs,
            's_idx': s_idx,
            'e_idx': e_idx,
            'signal_table': signal_table,
            'imf_stats_table': pd.DataFrame(imf_stats)
        }

    # ---- B. 创建双 GridSpec 架构画布 ----
    fig = plt.figure(figsize=(15.0, 10.5))

    # 顶部与下方网格之间留一条空白带，用显式边界固定在 0.73~0.81
    gs_top = fig.add_gridspec(
        nrows=1, ncols=1,
        left=0.10, right=0.98,
        bottom=0.81, top=0.92
    )

    gs_bottom = fig.add_gridspec(
        nrows=5, ncols=3,
        left=0.10, right=0.98,
        bottom=0.06, top=0.73,
        hspace=0.10,
        wspace=0.16
    )

    # 1. 顶部总览子图
    ax_overview = fig.add_subplot(gs_top[0, 0])

    # 2. 下方 5行 × 3列 分量网格
    axes_lower = np.empty((5, 3), dtype=object)
    for i in range(5):
        for j in range(3):
            axes_lower[i, j] = fig.add_subplot(gs_bottom[i, j])

    # ---- C. 绘制顶部总览图 ----
    overview_min_y, overview_max_y = np.inf, -np.inf

    for j, material_id in enumerate(MATERIAL_IDS):
        data = material_data[material_id]
        scheme = MATERIAL_COLOR_SCHEMES[material_id]

        # 归一化对比形态
        y_min, y_max = np.min(data['demand']), np.max(data['demand'])
        y_scaled = (data['demand'] - y_min) / (y_max - y_min + 1e-12)

        ax_overview.plot(
            data['dates'], y_scaled,
            color=scheme["orig"],
            linewidth=1.0,
            alpha=0.85,
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
            fill=False,
            edgecolor=scheme["orig"],
            linewidth=1.1,
            linestyle=(0, (4, 2.5)),
            alpha=0.9
        )
        ax_overview.add_patch(rect)

        # 虚线框悬浮文字标签
        ax_overview.text(
            win_start_dt + (win_end_dt - win_start_dt) / 2,
            1.05,
            f"展示视窗 [{material_id}]",
            color=scheme["orig"],
            fontsize=6.8,
            va='bottom', ha='center',
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.15',
                fc='white',
                ec=scheme["orig"],
                alpha=ANNOT_ALPHA,
                lw=ANNOT_LW
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
        fontsize=10.0,
        fontweight='bold',
        pad=10
    )
    ax_overview.legend(
        loc='upper left',
        ncol=3,
        framealpha=0.88,
        edgecolor='#cccccc',
        fontsize=7.2,
        columnspacing=1.0,
        borderpad=0.3
    )

    # ---- D. 绘制下方 5x3 截取时段分量图 ----
    for j, material_id in enumerate(MATERIAL_IDS):
        data = material_data[material_id]
        scheme = MATERIAL_COLOR_SCHEMES[material_id]
        color_orig = scheme["orig"]
        color_imfs = scheme["imfs"]

        win_dates = data['win_dates']
        win_demand = data['win_demand']
        win_imfs = data['win_imfs']
        imf_stats = data['imf_stats']

        # 将 Datetime 转为 numpy 数组用于 matplotlib 绘图
        x_dates = win_dates.to_numpy()

        # --- 1. 原始局部信号 (Row 0 in axes_lower) ---
        ax = axes_lower[0, j]
        ax.fill_between(x_dates, win_demand, color=color_orig, alpha=FILL_ALPHA_ORIG)
        ax.plot(x_dates, win_demand, color=color_orig, linewidth=1.0)
        stylize_axis(ax, show_x=False)
        # 列标题加粗，与顶部轴留出间距
        ax.set_title(f"{SUBFIG_TAGS[j]} {MATERIAL_NAMES[material_id]}", pad=8, fontweight='bold', fontsize=9.2)

        start_str_box = win_dates.iloc[0].strftime("%Y-%m")
        end_str_box = win_dates.iloc[-1].strftime("%Y-%m")
        ax.text(
            0.018, 0.88,
            f"窗口: {start_str_box} ~ {end_str_box}",
            transform=ax.transAxes,
            fontsize=6.8,
            va='top',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color_orig, alpha=ANNOT_ALPHA, lw=ANNOT_LW)
        )

        # --- 2. IMF1~4 局部信号 (Row 1~4 in axes_lower) ---
        for k in range(VMD_K):
            ax = axes_lower[k + 1, j]
            imf = win_imfs[k]
            ax.fill_between(x_dates, imf, color=color_imfs[k], alpha=FILL_ALPHA_IMF)
            ax.plot(x_dates, imf, color=color_imfs[k], linewidth=0.88)
            ax.axhline(0, color='#666666', linewidth=0.45, linestyle='-', alpha=0.5)
            stylize_axis(ax, show_x=(k == VMD_K - 1))

            p_val = imf_stats[k]['dominant_period_months']
            p_text = f"{p_val:.1f}月" if (p_val is not None and np.isfinite(p_val)) else "趋势"
            info = f"{IMF_LABELS_SHORT[k]} | Var={imf_stats[k]['var_pct']:.2f}% | T≈{p_text}"

            ax.text(
                0.018, 0.88, info,
                transform=ax.transAxes,
                fontsize=6.5,
                va='top',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color_imfs[k], alpha=ANNOT_ALPHA, lw=ANNOT_LW)
            )

        # --- 3. 底部 X 轴时间刻度精简格式化 ---
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

    # 每列底部 X 轴物理标签
    for j in range(3):
        axes_lower[-1, j].set_xlabel('时间', fontsize=8.2, fontweight='bold')

    # 公共行标签外挂，用偏移量避开 Y 轴刻度值
    add_row_labels(fig, axes_lower)

    # 论文大标题
    fig.suptitle(
        '不同形态电力物资需求序列的全生命周期 VMD 多尺度特征演化对比',
        fontsize=12.0,
        y=0.993,
        fontweight='bold'
    )

    # ---- E. 矩阵与数据落盘 ----
    fig_name = 'fig_vmd_combined_multi_material_final'

    export_data = {
        'figure_meta': {
            'figure_name': fig_name,
            'description': '三类物资全生命周期VMD计算，子图截取特征视窗展示（论文终稿标准）',
            'materials': MATERIAL_IDS,
            'vmd_k': VMD_K,
            'vmd_alpha': VMD_ALPHA,
            'vmd_tol': VMD_TOL,
        }
    }

    flat_export = {'figure_meta': export_data['figure_meta']}
    for material_id in MATERIAL_IDS:
        data = material_data[material_id]
        flat_export[f'{material_id}_meta'] = {
            'material_id': material_id,
            'material_name': MATERIAL_NAMES[material_id],
            'window_start': FEATURE_WINDOWS[material_id][0],
            'window_end': FEATURE_WINDOWS[material_id][1],
            'vmd_k': VMD_K,
            'vmd_alpha': VMD_ALPHA,
            'vmd_tol': VMD_TOL
        }
        flat_export[f'{material_id}_imf_stats'] = data['imf_stats']
        flat_export[f'{material_id}_reconstruction'] = data['reconstruction']
        flat_export[f'{material_id}_signal_table'] = data['signal_table']
        flat_export[f'{material_id}_imf_stats_table'] = data['imf_stats_table']

    save_figure_data(fig_name, flat_export)
    save_figure(fig, fig_name, formats=('png', 'pdf'))

    print('\n========== 论文终稿总图摘要 ==========')
    for material_id in MATERIAL_IDS:
        meta = flat_export[f'{material_id}_meta']
        recon = flat_export[f'{material_id}_reconstruction']
        print(
            f"[{material_id}] {meta['material_name']} | "
            f"全序列长度: {len(material_data[material_id]['demand'])} | "
            f"展示视窗: {meta['window_start']} ~ {meta['window_end']} | "
            f"重构度 R²={recon['R2']:.6f}, MAE={recon['MAE']:.4f}"
        )


# ---- 主程序入口 ----
if __name__ == '__main__':
    build_unified_vmd_visualization()
