"""
原始需求数据可视化（多物资自适应），画出时序全景、特征关联与季节性/ACF 三类图。

默认处理 MAT001，也可在命令行指定物资：
  python plot_raw_data_visualization.py
  python plot_raw_data_visualization.py MAT002
输出到 figures/raw_data_visualization/<物资>/。
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf, pacf

# ---- 多物资动态切换（须在导入 config 前完成） ----
# 支持命令行参数: python plot_raw_data_visualization.py MAT002
if len(sys.argv) > 1:
    os.environ["MATERIAL_SCENARIO"] = sys.argv[1].upper()

# 物资名称映射表 (用于动态标题)
MATERIAL_NAMES = {
    "001": "10kV油浸式变压器",
    "002": "架空绝缘导线",
    "003": "智能电表",
}
_CURRENT_MATERIAL = os.environ.get("MATERIAL_SCENARIO", "001").upper()
MATERIAL_NAME = MATERIAL_NAMES.get(_CURRENT_MATERIAL, "未知物资")

# 动态导入并重载 config，确保路径随 MATERIAL_SCENARIO 切换
import configs.config as cfg
import importlib
importlib.reload(cfg)

SIM_DATASET_PATH = cfg.SIM_DATASET_PATH
DATA_FIGURES_DIR = cfg.DATA_FIGURES_DIR
FIGURES_DIR = cfg.FIGURES_DIR
TRAIN_RATIO = cfg.TRAIN_RATIO
VAL_RATIO = cfg.VAL_RATIO
RANDOM_SEED = cfg.RANDOM_SEED
DEMAND_COL = cfg.DEMAND_COL

# 字体路径：若 config 中已删除字体变量，则使用系统默认
try:
    FONT_DEJAVUSANS = cfg.FONT_DEJAVUSANS_FILE_PATH
    FONT_SARASAMONOSC_REGULAR = cfg.FONT_SARASAMONOSC_REGULAR_FILE_PATH
    FONT_SARASAMONOSC_BOLD = cfg.FONT_SARASAMONOSC_BOLD_FILE_PATH
    HAS_CUSTOM_FONT = True
except AttributeError:
    HAS_CUSTOM_FONT = False

# ---- 项目路径 ----
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import (
    load_raw_data, clean_data_train, clean_data_with_stats, split_by_time
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
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.12,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'lines.antialiased': True,
    'patch.linewidth': 0.6,
    'font.weight': 'normal',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
})

# ---- 输出路径（按物资隔离） ----
SAVE_DIR = os.path.join(FIGURES_DIR, 'raw_data_visualization', _CURRENT_MATERIAL.lower())
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- 配色方案 ----
COLOR_DEMAND = '#2C3E50'
COLOR_TREND = '#E74C3C'
COLOR_SEASON = '#2980B9'
COLOR_SPLIT = '#7F8C8D'
COLOR_ACCENT = '#E67E22'
COLOR_BOX = '#3498DB'

EXTERNAL_COLS = [
    'temperature', 'extreme_weather_days', 'wind_speed',
    'macro_invest_growth', 'gdp_growth', 'copper_price_index',
    'power_load_MW', 'equipment_aging_rate', 'bidding_intensity',
    'renewable_capacity_MW', 'is_spring_festival',
]
EXTERNAL_LABELS_CN = [
    '温度', '极端天气', '风速', '投资增速', 'GDP增速', '铜价',
    '负荷', '老化率', '招标强度', '新能源', '春节',
]

# 全24维特征元数据（与主模型 ConsensusFeatureSelector 一致）
ALL_FEATURE_NAMES = [
    'temperature', 'extreme_weather_days', 'wind_speed',
    'macro_invest_growth', 'gdp_growth', 'copper_price_index',
    'power_load_MW', 'equipment_aging_rate', 'bidding_intensity',
    'renewable_capacity_MW', 'is_spring_festival',
    'month_sin', 'month_cos', 'quarter',
    'IMF1', 'IMF2', 'IMF3', 'IMF4',
    'delta_IMF1', 'delta_IMF2', 'delta_IMF3', 'delta_IMF4',
    'delta_demand', 'demand',
]
ALL_FEATURE_LABELS_CN = [
    '温度', '极端天气', '风速', '投资增速', 'GDP增速', '铜价',
    '负荷', '老化率', '招标强度', '新能源', '春节',
    '月sin', '月cos', '季度',
    'IMF1', 'IMF2', 'IMF3', 'IMF4',
    'ΔIMF1', 'ΔIMF2', 'ΔIMF3', 'ΔIMF4',
    'Δ需求', '需求',
]
ALL_FEATURE_GROUPS = [
    '气象', '气象', '气象', '经济', '经济', '经济',
    '电力', '电力', '电力', '电力', '时间',
    '时间', '时间', '时间',
    'VMD-IMF', 'VMD-IMF', 'VMD-IMF', 'VMD-IMF',
    '动量', '动量', '动量', '动量', '动量', '目标',
]
FEATURE_GROUP_COLORS = {
    '气象': '#E74C3C', '经济': '#2980B9', '电力': '#27AE60',
    '时间': '#F39C12', 'VMD-IMF': '#8E44AD', '动量': '#1ABC9C',
    '目标': '#2C3E50',
}
GROUP_ORDER = ['气象', '经济', '电力', '时间', 'VMD-IMF', '动量', '目标']

VMD_K = 4
VMD_ALPHA = 2000


# ---- 数据加载 ----

def load_raw_demand():
    """加载需求序列与外部特征"""
    df = load_raw_data(SIM_DATASET_PATH)
    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)
    train_d = train_clean[DEMAND_COL].values
    val_d = val_clean[DEMAND_COL].values
    test_d = test_clean[DEMAND_COL].values
    full_d = np.concatenate([train_d, val_d, test_d])
    dates = pd.concat([train_clean, val_clean, test_clean]).index
    full_df = pd.concat([train_clean, val_clean, test_clean])
    return full_d, dates, len(train_d), len(train_d) + len(val_d), full_df


def load_full_feature_matrix():
    """加载全24维特征矩阵"""
    df = load_raw_data(SIM_DATASET_PATH)
    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    train_demand = train_clean[DEMAND_COL].values
    val_demand = val_clean[DEMAND_COL].values
    test_demand = test_clean[DEMAND_COL].values
    full_demand = np.concatenate([train_demand, val_demand, test_demand])
    dates = pd.concat([train_clean, val_clean, test_clean]).index

    # VMD分解
    cache_file = os.path.join(
        DATA_FIGURES_DIR, f'vmd_cache_K{VMD_K}_alpha{VMD_ALPHA}.npz')
    imfs_full = _vmd_decompose(full_demand, K=VMD_K, alpha=VMD_ALPHA,
                                cache_path=cache_file)

    for d in [train_clean, val_clean, test_clean]:
        d['month_sin'] = np.sin(2 * np.pi * d.index.month / 12)
        d['month_cos'] = np.cos(2 * np.pi * d.index.month / 12)
        d['quarter'] = d.index.quarter

    feature_cols = EXTERNAL_COLS + ['month_sin', 'month_cos', 'quarter']
    lengths = [len(train_clean), len(val_clean), len(test_clean)]

    X_base = train_clean[feature_cols].values
    imf_segs = [imfs_full[k][:lengths[0]] for k in range(VMD_K)]
    demand_seg = train_demand

    imf_cols = np.column_stack(imf_segs)
    delta_imf_cols = np.zeros_like(imf_cols)
    delta_imf_cols[1:] = imf_cols[1:] - imf_cols[:-1]
    delta_demand = np.zeros_like(demand_seg)
    delta_demand[1:] = demand_seg[1:] - demand_seg[:-1]

    X_train_raw = np.column_stack(
        [X_base, imf_cols, delta_imf_cols, delta_demand, demand_seg])

    X_for_corr = X_train_raw[:-1]
    y_for_corr = train_demand[1:]

    return X_for_corr, y_for_corr, dates, len(train_demand)


def _vmd_decompose(signal, K=VMD_K, alpha=VMD_ALPHA, cache_path=None):
    try:
        from vmdpy import VMD
    except ImportError:
        raise ImportError("请安装 vmdpy: pip install vmdpy")
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        u = data["u"]
    else:
        u, u_hat, omega = VMD(signal, alpha, 0.0, K, 0, 1, 1e-7)
        if cache_path:
            np.savez_compressed(cache_path, u=u, u_hat=u_hat, omega=omega)
    n_in, n_out = len(signal), u.shape[1]
    if n_out < n_in:
        pad = np.zeros((u.shape[0], n_in - n_out))
        for k in range(u.shape[0]):
            if n_out >= 2:
                slope = u[k, -1] - u[k, -2]
                pad[k] = u[k, -1] + slope * np.arange(1, n_in - n_out + 1)
            else:
                pad[k] = u[k, -1]
        u = np.hstack([u, pad])
    residual = signal - u.sum(axis=0)
    imf_vars = np.array([np.var(u[k]) for k in range(K)])
    total_var = imf_vars.sum()
    ratios = imf_vars / total_var if total_var > 0 else np.ones(K) / K
    u_corrected = u.copy()
    for k in range(K):
        u_corrected[k] += residual * ratios[k]
    final_res = signal - u_corrected.sum(axis=0)
    u_corrected += final_res / K
    return u_corrected


# ---- 通用工具 ----

def save_figure(fig, name, formats=('png', 'pdf')):
    paths = []
    for fmt in formats:
        p = os.path.join(SAVE_DIR, f'{name}.{fmt}')
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    suffix = ' + PDF' if len(paths) > 1 else ''
    print(f'  ✅ 已保存: {paths[0]}{suffix}')
    return paths[0]

def safe_float(v):
    if isinstance(v, np.ndarray):
        v = np.squeeze(v)
        if v.ndim == 0:
            return float(v)
        else:
            return float(v.mean())
    return float(v)

def save_figure_data(fig_name, data_dict):
    json_path = os.path.join(SAVE_DIR, f'{fig_name}_data.json')
    xlsx_path = os.path.join(SAVE_DIR, f'{fig_name}_data.xlsx')
    json_safe = {}
    for key, val in data_dict.items():
        if isinstance(val, pd.DataFrame):
            json_safe[key] = val.to_dict(orient='list')
        elif isinstance(val, np.ndarray):
            json_safe[key] = val.tolist()
        elif isinstance(val, (np.floating, np.integer)):
            json_safe[key] = float(val)
        elif isinstance(val, dict):
            json_safe[key] = val
        else:
            json_safe[key] = str(val)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_safe, f, indent=2, ensure_ascii=False)
    print(f'  📊 数据JSON已保存: {json_path}')
    csv_frames = {}
    for key, val in data_dict.items():
        if isinstance(val, pd.DataFrame):
            csv_frames[key] = val
        elif isinstance(val, np.ndarray) and val.ndim == 2:
            csv_frames[key] = pd.DataFrame(val)
    if csv_frames:
        with pd.ExcelWriter(xlsx_path) as writer:
            for sheet_name, df in csv_frames.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f'  📊 数据Excel已保存: {xlsx_path}')

def add_split_lines(ax, n_train, n_trainval, total_len):
    for pos in [n_train, n_trainval]:
        ax.axvline(pos, color=COLOR_SPLIT, ls=':', lw=0.7, alpha=0.6, zorder=1)
    y_lo, y_hi = ax.get_ylim()
    y_pos = y_hi - (y_hi - y_lo) * 0.06
    for start, end, name in [(0, n_train, 'Train'),
                              (n_train, n_trainval, 'Val'),
                              (n_trainval, total_len, 'Test')]:
        ax.text((start + end) / 2, y_pos, name, ha='center',
                fontsize=8, color=COLOR_SPLIT, fontstyle='italic', alpha=0.7)

def add_year_ticks(ax, dates, step=3):
    years = dates.year
    unique = np.arange(years.min(), years.max() + 1, step)
    positions, labels = [], []
    for yr in unique:
        idx = np.where(years == yr)[0]
        if len(idx) > 0:
            positions.append(idx[0])
            labels.append(str(yr))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right')


# ---- 图1: 需求时序全景图 ----
def plot_demand_overview(demand, dates, n_train, n_trainval, material_name):
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, figure=fig, hspace=0.30,
                  top=0.93, bottom=0.08, left=0.09, right=0.96,
                  height_ratios=[2.5, 1])

    x = np.arange(len(demand))
    n = len(demand)
    window = 12

    # ---- (a) 原始需求 + 趋势 + 波动带 ----
    ax_a = fig.add_subplot(gs[0])
    roll_mean = pd.Series(demand).rolling(window, center=True,
                                          min_periods=1).mean().values
    roll_std = pd.Series(demand).rolling(window, center=True,
                                         min_periods=1).std().values
    roll_std = np.nan_to_num(roll_std, nan=0)

    ax_a.fill_between(x, roll_mean - roll_std, roll_mean + roll_std,
                      alpha=0.15, color=COLOR_SEASON, label='1σ波动带')
    ax_a.plot(x, roll_mean, color=COLOR_TREND, lw=1.5, alpha=0.8,
              label=f'{window}月滚动均值(趋势)')
    ax_a.plot(x, demand, color=COLOR_DEMAND, lw=0.5, alpha=0.6,
              label='月度需求')

    months_arr = dates.month
    month_means = np.array([demand[months_arr == m].mean()
                            for m in range(1, 13)])
    seasonal_template = np.array([month_means[m - 1] for m in months_arr])
    ax_a.plot(x, seasonal_template, color=COLOR_SEASON, lw=1.0, ls='--',
              alpha=0.5, label='年内季节性模板')

    ax_a.set_ylabel('需求量', fontsize=12)
    # 标题使用传入的物资名称
    ax_a.set_title(f'{material_name}月度需求时序全景', fontsize=14, pad=8)
    ax_a.legend(loc='upper right', fontsize=8, framealpha=0.85, ncol=2)
    ax_a.grid(True, alpha=0.15, ls='--')
    ax_a.set_xlim(0, n - 1)
    add_split_lines(ax_a, n_train, n_trainval, n)

    cv = demand.std() / demand.mean() * 100
    info = (f'N={n}月 | μ={demand.mean():.0f} | σ={demand.std():.0f}\n'
            f'CV={cv:.1f}% | Range=[{demand.min():.0f}, {demand.max():.0f}]')
    ax_a.text(0.005, 0.05, info, transform=ax_a.transAxes, fontsize=8,
              va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                       ec='gray', alpha=0.9, lw=0.5))

    # inset: 去趋势残差
    ax_ins = ax_a.inset_axes([0.60, 0.04, 0.38, 0.28])
    detrended = demand - roll_mean
    ax_ins.plot(x, detrended, color='#8E44AD', lw=0.4, alpha=0.7)
    ax_ins.axhline(0, color='gray', lw=0.3)
    ax_ins.fill_between(x, detrended, alpha=0.15, color='#8E44AD')
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.text(0.50, 0.92, '去趋势残差 → VMD分解目标',
                transform=ax_ins.transAxes, fontsize=7, ha='center',
                color='#8E44AD', fontweight='bold')
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color('#999')

    # ---- (b) 月度环比 + 滚动波动性 ----
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    delta = np.zeros(n)
    delta[1:] = demand[1:] - demand[:-1]

    colors_delta = ['#27AE60' if d >= 0 else '#E74C3C' for d in delta]
    ax_b.bar(x, delta, color=colors_delta, alpha=0.35, width=1.0,
             label='月度环比Δ')

    roll_vol = pd.Series(np.abs(delta)).rolling(
        window, center=True, min_periods=1).mean().values
    ax_b.plot(x, roll_vol, color=COLOR_TREND, lw=1.2, alpha=0.8,
              label=f'波动性({window}月滚动|Δ|均值)')
    ax_b.axhline(0, color='gray', lw=0.4, alpha=0.5)
    ax_b.set_ylabel('Δ需求量', fontsize=11)
    ax_b.set_xlabel('时间 (月)', fontsize=11)
    ax_b.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax_b.grid(True, alpha=0.12, ls='--')
    add_year_ticks(ax_b, dates)

    avg_vol = np.mean(np.abs(delta))
    max_vol = np.max(np.abs(delta))
    ax_b.text(0.005, 0.05,
              f'平均波动: {avg_vol:.0f}/月 | 最大波动: {max_vol:.0f}',
              transform=ax_b.transAxes, fontsize=8, va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow',
                       ec='gray', alpha=0.85, lw=0.4))

    # ---- 数据导出 ----
    fig_overview_data = {
        'demand_stats': {
            'N': int(n), 'mean': float(demand.mean()), 'std': float(demand.std()),
            'CV_pct': round(float(cv), 1),
            'min': float(demand.min()), 'max': float(demand.max()),
            'avg_monthly_delta': float(np.mean(np.abs(delta))),
            'max_monthly_delta': float(np.max(np.abs(delta))),
        },
    }
    save_figure_data('fig_demand_overview', fig_overview_data)

    return save_figure(fig, 'fig_demand_overview')


# ---- 图2: 需求与特征关联分析 ----
def plot_feature_correlation(demand, full_df, n_train,
                             X_full=None, y_full=None,
                             fs_state=None, material_name=''):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, figure=fig, hspace=0.28,
                  top=0.93, bottom=0.06, left=0.12, right=0.96,
                  height_ratios=[0.8, 1.2])

    train_df = full_df.iloc[:n_train]

    # ---- (a) 外部特征与需求的关联强度 ----
    ax_a = fig.add_subplot(gs[0])
    n_features = len(EXTERNAL_COLS)

    sp_corrs = []
    mi_vals = []
    for col in EXTERNAL_COLS:
        xi = train_df[col].values
        try:
            corr, _ = spearmanr(xi, demand[:n_train])
            sp_corrs.append(abs(corr) if not np.isnan(corr) else 0.0)
        except Exception:
            sp_corrs.append(0.0)
        try:
            mi = mutual_info_regression(
                xi.reshape(-1, 1), demand[:n_train],
                n_neighbors=5, random_state=42)[0]
            mi_vals.append(mi)
        except Exception:
            mi_vals.append(0.0)

    mi_max = max(mi_vals) if max(mi_vals) > 0 else 1
    mi_norm = [v / mi_max for v in mi_vals]

    x_pos = np.arange(n_features)
    width = 0.35

    group_map = {
        'temperature': '气象', 'extreme_weather_days': '气象',
        'wind_speed': '气象', 'macro_invest_growth': '经济',
        'gdp_growth': '经济', 'copper_price_index': '经济',
        'power_load_MW': '电力', 'equipment_aging_rate': '电力',
        'bidding_intensity': '电力', 'renewable_capacity_MW': '电力',
        'is_spring_festival': '时间',
    }
    bar_colors = [FEATURE_GROUP_COLORS.get(group_map.get(c, ''), '#888')
                  for c in EXTERNAL_COLS]

    ax_a.bar(x_pos - width / 2, sp_corrs, width, color=bar_colors,
             alpha=0.7, edgecolor='white', lw=0.6, label='Spearman |ρ|')
    ax_a.bar(x_pos + width / 2, mi_norm, width, color=bar_colors,
             alpha=0.35, edgecolor=bar_colors, lw=1.0,
             label='MI (归一化)')

    ax_a.axhline(0.05, color='gray', ls='--', lw=0.6, alpha=0.6)
    ax_a.fill_between([-0.5, n_features - 0.5], 0, 0.05,
                      alpha=0.06, color='red')
    ax_a.text(n_features - 0.5, 0.025, '|ρ|<0.05 预筛选移除区',
              fontsize=7, color='#C0392B', ha='right', fontstyle='italic')

    for i in range(n_features):
        if sp_corrs[i] > 0.05:
            ax_a.text(i - width / 2, sp_corrs[i] + 0.015,
                      f'{sp_corrs[i]:.2f}', ha='center', fontsize=7,
                      color=bar_colors[i], fontweight='bold')
        else:
            ax_a.text(i - width / 2, sp_corrs[i] + 0.015,
                      f'{sp_corrs[i]:.2f}', ha='center', fontsize=6,
                      color='#999')

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(EXTERNAL_LABELS_CN, fontsize=9, rotation=30,
                         ha='right')
    ax_a.set_ylabel('关联强度', fontsize=11)
    ax_a.set_title(f'(a) {material_name}外部特征与需求的关联强度 (训练集)', fontsize=12)

    metric_handles = [
        Patch(facecolor='#888', alpha=0.7, label='Spearman |ρ|'),
        Patch(facecolor='#888', alpha=0.35, edgecolor='#888',
              linewidth=1.0, label='MI (归一化)'),
    ]
    group_patches = [
        Patch(facecolor=c, alpha=0.6, label=l)
        for l, c in FEATURE_GROUP_COLORS.items()
        if l in ['气象', '经济', '电力', '时间']
    ]
    ax_a.legend(handles=metric_handles + group_patches,
                loc='upper right', fontsize=7, framealpha=0.85,
                ncol=3, handlelength=1.2)
    ax_a.grid(True, alpha=0.12, ls='--', axis='y')
    ax_a.set_ylim(0, max(max(sp_corrs), max(mi_norm)) * 1.25)
    ax_a.set_xlim(-0.5, n_features - 0.5)

    # ---- (b) 全24维特征间相关性热力图 ----
    ax_b = fig.add_subplot(gs[1])

    if X_full is not None and y_full is not None:
        n_all = X_full.shape[1] + 1
        all_data = np.column_stack([X_full, y_full])
        all_labels = ALL_FEATURE_LABELS_CN + ['目标(t+1)']
        all_groups = ALL_FEATURE_GROUPS + ['目标']
    else:
        corr_data = train_df[EXTERNAL_COLS].copy()
        corr_data['demand'] = demand[:n_train]
        all_cols = EXTERNAL_COLS + ['demand']
        all_labels = EXTERNAL_LABELS_CN + ['需求']
        all_groups = [group_map.get(c, '目标') for c in all_cols]
        all_data = corr_data[all_cols].values
        n_all = len(all_labels)

    corr_matrix = np.zeros((n_all, n_all))
    for i in range(n_all):
        for j in range(i, n_all):
            try:
                c, _ = spearmanr(all_data[:, i], all_data[:, j])
                corr_matrix[i, j] = c if not np.isnan(c) else 0.0
                corr_matrix[j, i] = corr_matrix[i, j]
            except Exception:
                corr_matrix[i, j] = 0.0
                corr_matrix[j, i] = 0.0

    im = ax_b.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                     aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.025, pad=0.02)
    cbar.set_label('Spearman ρ', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for i in range(n_all):
        for j in range(n_all):
            val = corr_matrix[i, j]
            is_important = (i == j or abs(val) > 0.7 or
                            all_groups[i] == '目标' or
                            all_groups[j] == '目标')
            color = 'white' if abs(val) > 0.55 else '#333'
            fs = 6 if is_important else 4
            fw = 'bold' if is_important else 'normal'
            text = '1.00' if i == j else f'{val:.2f}'
            ax_b.text(j, i, text, ha='center', va='center',
                      fontsize=fs, color=color, fontweight=fw)

    target_col = n_all - 1
    ax_b.add_patch(Rectangle((target_col - 0.5, -0.5), 1, n_all,
                             fill=False, edgecolor=COLOR_TREND,
                             lw=2.0, zorder=5))
    ax_b.add_patch(Rectangle((-0.5, target_col - 0.5), n_all, 1,
                             fill=False, edgecolor=COLOR_TREND,
                             lw=2.0, zorder=5))

    removed_red = set()
    removed_corr_set = set()
    n_redundancy_actual = 0

    if fs_state is not None:
        removed_red = set(fs_state.get('removed_by_redundancy', []))
        removed_corr_set = set(fs_state.get('removed_by_corr', []))
        n_redundancy_actual = len(removed_red)

        surviving = set(range(n_all - 1)) - removed_corr_set
        for idx in sorted(removed_red):
            if idx >= n_all - 1:
                continue
            best_j = -1
            best_abs = 0
            for j in range(n_all - 1):
                if j == idx or j in removed_red or j in removed_corr_set:
                    continue
                abs_c = abs(corr_matrix[idx, j])
                if abs_c > best_abs:
                    best_abs = abs_c
                    best_j = j
            if best_j >= 0 and best_abs > 0.85:
                ax_b.add_patch(Rectangle(
                    (best_j - 0.5, idx - 0.5), 1, 1,
                    fill=False, edgecolor=COLOR_TREND,
                    lw=1.8, ls='--', zorder=4))
                if best_j > idx:
                    ax_b.add_patch(Rectangle(
                        (idx - 0.5, best_j - 0.5), 1, 1,
                        fill=False, edgecolor=COLOR_TREND,
                        lw=1.2, ls='--', alpha=0.5, zorder=4))

        for idx in sorted(removed_corr_set):
            if idx < n_all - 1:
                ax_b.add_patch(Rectangle(
                    (idx - 0.5, -0.5), 1, n_all,
                    fill=True, facecolor='#7F8C8D', alpha=0.15, zorder=2))
                ax_b.add_patch(Rectangle(
                    (-0.5, idx - 0.5), n_all, 1,
                    fill=True, facecolor='#7F8C8D', alpha=0.15, zorder=2))
    else:
        for i in range(n_all - 1):
            for j in range(i + 1, n_all - 1):
                if abs(corr_matrix[i, j]) > 0.85:
                    ax_b.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor=COLOR_TREND,
                        lw=1.5, ls='--', zorder=4))
                    n_redundancy_actual += 1

    prev_group = all_groups[0]
    for i in range(1, n_all):
        if all_groups[i] != prev_group:
            ax_b.axhline(i - 0.5, color='white', lw=1.5, zorder=3)
            ax_b.axvline(i - 0.5, color='white', lw=1.5, zorder=3)
            prev_group = all_groups[i]

    group_start = 0
    prev_group = all_groups[0]
    for i in range(1, n_all + 1):
        cur_group = all_groups[i] if i < n_all else None
        if cur_group != prev_group:
            color = FEATURE_GROUP_COLORS.get(prev_group, '#888')
            ax_b.add_patch(Rectangle(
                (-1.0, group_start - 0.5), 0.5, i - group_start,
                facecolor=color, alpha=0.6, edgecolor='white', lw=0.5,
                clip_on=False, zorder=6))
            mid = (group_start + i - 1) / 2
            ax_b.text(-1.3, mid, prev_group, ha='right', va='center',
                      fontsize=7, color=color, fontweight='bold',
                      clip_on=False)
            group_start = i
            prev_group = cur_group

    ax_b.set_xticks(range(n_all))
    ax_b.set_xticklabels(all_labels, fontsize=6, rotation=55, ha='right')
    ax_b.set_yticks(range(n_all))
    ax_b.set_yticklabels(all_labels, fontsize=6)

    title_suffix = f' ({n_redundancy_actual}对冗余)' \
        if n_redundancy_actual > 0 else ''
    source_note = ' [主模型真实结果]' if fs_state is not None else ''
    ax_b.set_title(
        f'(b) {material_name}全24维特征间Spearman相关性{title_suffix}{source_note}\n'
        f'红框=Step3冗余对(|ρ|>0.85), 灰带=Step1移除, 红线=目标列',
        fontsize=12)

    ax_b.set_xticks(np.arange(-0.5, n_all, 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, n_all, 1), minor=True)
    ax_b.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax_b.tick_params(which='minor', size=0)

    df_ext_corr = pd.DataFrame({
        'feature_cn': EXTERNAL_LABELS_CN,
        'feature_en': EXTERNAL_COLS,
        'spearman_abs': [round(float(v), 4) for v in sp_corrs],
        'mi_norm': [round(float(v), 4) for v in mi_norm],
        'group': [group_map.get(c, '') for c in EXTERNAL_COLS],
    })

    fig_corr_data = {
        'external_feature_correlations': df_ext_corr,
        'correlation_matrix_shape': list(corr_matrix.shape),
        'n_redundancy_pairs': n_redundancy_actual,
    }
    if fs_state is not None:
        fig_corr_data['feature_selector_state'] = {
            'removed_by_corr': fs_state.get('removed_by_corr', []),
            'removed_by_redundancy': fs_state.get('removed_by_redundancy', []),
            'selected_indices': fs_state.get('selected_indices', []),
        }
    save_figure_data('fig_feature_correlation', fig_corr_data)

    return save_figure(fig, 'fig_feature_correlation')


# ---- 图3: 需求季节性规律与时间依赖结构 ----
def plot_seasonality_and_acf(demand, dates, n_train, n_trainval, material_name):
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30,
                  top=0.93, bottom=0.10, left=0.08, right=0.96)

    # ---- (a) 按月箱线图 ----
    ax_a = fig.add_subplot(gs[0])
    months = dates.month
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                   '7月', '8月', '9月', '10月', '11月', '12月']
    demand_by_month = [demand[months == m] for m in range(1, 13)]

    bp = ax_a.boxplot(demand_by_month, patch_artist=True, widths=0.6,
                      showfliers=True,
                      flierprops=dict(marker='o', ms=2.5, alpha=0.4))

    month_colors = [
        '#5DADE2', '#5DADE2', '#58D68D',
        '#58D68D', '#F4D03F', '#E74C3C',
        '#E74C3C', '#E74C3C', '#F4D03F',
        '#58D68D', '#5DADE2', '#5DADE2',
    ]
    for patch, color in zip(bp['boxes'], month_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor('#555')
    for line in bp['medians']:
        line.set_color('#2C3E50')
        line.set_linewidth(1.2)

    medians = [np.median(d) for d in demand_by_month]
    ax_a.plot(range(1, 13), medians, color=COLOR_TREND, lw=1.5,
              marker='o', ms=4, alpha=0.8, label='月中位数', zorder=5)

    season_spans = [
        (0.5, 3.5, '#5DADE2', 0.04, '冬'),
        (3.5, 6.5, '#58D68D', 0.04, '春'),
        (6.5, 9.5, '#E74C3C', 0.04, '夏'),
        (9.5, 12.5, '#F4D03F', 0.04, '秋'),
    ]
    yl = ax_a.get_ylim()
    for left, right, color, alpha, label in season_spans:
        ax_a.axvspan(left, right, alpha=alpha, color=color, zorder=0)
        ax_a.text((left + right) / 2, yl[1] - (yl[1] - yl[0]) * 0.03,
                  label, ha='center', fontsize=7, color=color,
                  fontweight='bold', alpha=0.6)

    ax_a.set_xticklabels(month_names, fontsize=9)
    ax_a.set_ylabel('需求量', fontsize=11)
    ax_a.set_title(f'(a) {material_name}需求月度分布', fontsize=12)
    ax_a.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax_a.grid(True, alpha=0.15, ls='--', axis='y')

    peak_month = int(np.argmax(medians)) + 1
    valley_month = int(np.argmin(medians)) + 1
    seasonal_range = max(medians) - min(medians)
    cv_seasonal = seasonal_range / demand.mean() * 100

    ax_a.annotate(f'峰值({peak_month}月)',
                  xy=(peak_month, medians[peak_month - 1]),
                  xytext=(peak_month + 1.2,
                          medians[peak_month - 1] + seasonal_range * 0.08),
                  fontsize=8, color='#E74C3C', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=0.8))
    ax_a.annotate(f'谷值({valley_month}月)',
                  xy=(valley_month, medians[valley_month - 1]),
                  xytext=(valley_month + 1.5,
                          medians[valley_month - 1] - seasonal_range * 0.05),
                  fontsize=8, color='#5DADE2', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#5DADE2', lw=0.8))

    ax_a.text(0.02, 0.05,
              f'峰谷差: {seasonal_range:.0f} (占均值{cv_seasonal:.1f}%)\n'
              f'CV(月间): {np.std(medians)/np.mean(medians)*100:.1f}%',
              transform=ax_a.transAxes, fontsize=8, va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow',
                       ec='gray', alpha=0.9, lw=0.5))

    # ---- (b) ACF + PACF ----
    ax_b = fig.add_subplot(gs[1])
    train_demand = demand[:n_train]
    nlags = 36
    acf_vals = acf(train_demand, nlags=nlags, fft=True)
    pacf_vals = pacf(train_demand, nlags=nlags)
    lags = np.arange(len(acf_vals))
    lags_p = np.arange(len(pacf_vals))

    ax_b.bar(lags - 0.15, acf_vals, width=0.3, color=COLOR_SEASON,
             alpha=0.7, label='ACF')
    ax_b.bar(lags_p + 0.15, pacf_vals, width=0.3, color=COLOR_TREND,
             alpha=0.7, label='PACF')

    n_tr = len(train_demand)
    conf = 1.96 / np.sqrt(n_tr)
    ax_b.axhline(conf, color='gray', ls='--', lw=0.5, alpha=0.6)
    ax_b.axhline(-conf, color='gray', ls='--', lw=0.5, alpha=0.6)
    ax_b.fill_between(lags, -conf, conf, alpha=0.05, color='gray')
    ax_b.text(nlags - 0.5, conf + 0.02, '95%置信', fontsize=7,
              color='gray', ha='right')

    for lag in [12, 24]:
        if lag < len(acf_vals):
            ax_b.axvline(lag, color=COLOR_SEASON, ls=':', lw=0.8, alpha=0.4)
            ax_b.plot(lag, acf_vals[lag], 'o', color=COLOR_SEASON, ms=6,
                      zorder=5, markeredgecolor='white', markeredgewidth=0.5)
            ax_b.text(lag + 0.5, acf_vals[lag] + 0.025,
                      f'lag={lag}\nACF={acf_vals[lag]:.3f}',
                      fontsize=7, color=COLOR_SEASON, fontweight='bold')

    ax_b.plot(1, acf_vals[1], 'o', color=COLOR_TREND, ms=5, zorder=5)
    ax_b.text(1.5, acf_vals[1] + 0.025,
              f'lag=1\nACF={acf_vals[1]:.3f}',
              fontsize=7, color=COLOR_TREND)
    ax_b.axhline(0, color='gray', lw=0.3, alpha=0.5)
    ax_b.set_xlabel('滞后阶数 (月)', fontsize=11)
    ax_b.set_ylabel('自相关系数', fontsize=11)
    ax_b.set_title(f'(b) {material_name}ACF/PACF (训练集)', fontsize=12)
    ax_b.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax_b.grid(True, alpha=0.12, ls='--')
    ax_b.set_xlim(-0.5, nlags + 0.5)

    ax_b.text(0.02, 0.05,
              'lag=1显著 → 短期自回归依赖\n'
              'lag=12,24显著 → 年周期性\n'
              'PACF快速衰减 → BiLSTM捕获长程依赖',
              transform=ax_b.transAxes, fontsize=8, va='bottom',
              color='#333',
              bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0',
                       ec='gray', alpha=0.92, lw=0.5))

    monthly_stats = []
    for m in range(1, 13):
        d_m = demand[months == m]
        monthly_stats.append({
            'month': m, 'mean': float(d_m.mean()), 'median': float(np.median(d_m)),
            'std': float(d_m.std()), 'count': int(len(d_m)),
        })

    nlags_export = min(nlags, len(acf_vals) - 1)
    fig_season_data = {
        'monthly_stats': monthly_stats,
        'acf': {f'lag_{i}': round(float(acf_vals[i]), 4) for i in range(nlags_export + 1)},
        'pacf': {f'lag_{i}': round(float(pacf_vals[i]), 4) for i in range(len(pacf_vals))},
        'seasonality_summary': {
            'peak_month': int(peak_month), 'valley_month': int(valley_month),
            'seasonal_range': float(seasonal_range),
            'cv_seasonal_pct': round(float(cv_seasonal), 1),
        },
    }
    save_figure_data('fig_seasonality_acf', fig_season_data)

    return save_figure(fig, 'fig_seasonality_acf')


# ---- 附: 纯净原始信号图 ----
def plot_clean_original_signal(demand, material_name):
    fig, ax = plt.subplots(figsize=(7, 1.6))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.06)
    ax.plot(np.arange(len(demand)), demand, color=COLOR_DEMAND, linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#888')
    ax.tick_params(length=0)
    ax.axhline(demand.mean(), color='#ccc', ls=':', lw=0.3)
    return save_figure(fig, 'fig_original_signal_clean', formats=('png',))


# ---- 主函数 ----
def main():
    print('=' * 60)
    print(f'  原始数据可视化 — 物资: {_CURRENT_MATERIAL} ({MATERIAL_NAME})')
    print(f'  输出目录: {SAVE_DIR}')
    print(f'  数据源: {SIM_DATASET_PATH}')
    print('=' * 60)

    print('\n[1/7] 加载原始需求数据...')
    demand, dates, n_train, n_trainval, full_df = load_raw_demand()
    T = len(demand)
    print(f'  信号长度: {T} | 训练集: {n_train} | 训+验证: {n_trainval}')

    print('\n[2/7] 加载全24维特征矩阵 (VMD分解)...')
    try:
        X_full, y_full, dates_full, n_train_full = load_full_feature_matrix()
        print(f'  特征矩阵: {X_full.shape}')
    except Exception as e:
        print(f'  ⚠️ 全特征矩阵加载失败: {e}')
        X_full, y_full = None, None

    print('\n[3/7] 加载主模型筛选器状态...')
    fs_state_path = os.path.join(
        DATA_FIGURES_DIR, 'feature_selector_state.json')
    fs_state = None
    if os.path.exists(fs_state_path):
        with open(fs_state_path, 'r', encoding='utf-8') as f:
            fs_state = json.load(f)
        fs_state['removed_by_redundancy'] = [
            int(x) for x in fs_state.get('removed_by_redundancy', [])]
        fs_state['removed_by_corr'] = [
            int(x) for x in fs_state.get('removed_by_corr', [])]
        n_red = len(fs_state['removed_by_redundancy'])
        n_corr = len(fs_state['removed_by_corr'])
        print(f'  Step1移除: {n_corr}个 | Step3冗余删除: {n_red}个')
    else:
        print(f'  ⚠️ 未找到 {fs_state_path}')
        print('  图2(b)将回退为自行计算冗余对 (可能不精确)')

    print('\n[4/7] 图1: 需求时序全景图...')
    p1 = plot_demand_overview(demand, dates, n_train, n_trainval, MATERIAL_NAME)

    print('\n[5/7] 图2: 特征关联分析 (全24维, JSON真实冗余对)...')
    p2 = plot_feature_correlation(demand, full_df, n_train,
                                  X_full, y_full, fs_state, MATERIAL_NAME)

    print('\n[6/7] 图3: 季节性规律与时间依赖结构...')
    p3 = plot_seasonality_and_acf(demand, dates, n_train, n_trainval, MATERIAL_NAME)

    print('\n[7/7] 纯净原始信号图 (架构图素材)...')
    p_clean = plot_clean_original_signal(demand, MATERIAL_NAME)

    print('\n' + '=' * 60)
    print('  ✅ 全部原始数据可视化图已生成!')
    print(f'  图1 (需求时序全景): {p1}')
    print(f'  图2 (特征关联分析): {p2}')
    print(f'  图3 (季节性与依赖): {p3}')
    print(f'  架构图素材: {p_clean}')
    print('=' * 60)


if __name__ == '__main__':
    main()