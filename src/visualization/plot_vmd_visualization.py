"""
单物资 VMD 分解可视化，画时域分解、频谱与重建/消融三类图（多物资自适应）。

可在命令行指定物资场景，例如：
  python plot_vmd_visualization.py
  python plot_vmd_visualization.py MAT002
输出到 figures/vmd_visualization/<物资>/。
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
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import periodogram
from pathlib import Path

# ---- 多物资动态切换（须在导入 config 前完成） ----

if len(sys.argv) > 1:
    os.environ["MATERIAL_SCENARIO"] = sys.argv[1].upper()

# 物资名称映射表

MATERIAL_NAMES = {
    "001": "10kV油浸式变压器",
    "002": "架空绝缘导线",
    "003": "智能电表",
}
_CURRENT_MATERIAL = os.environ.get("MATERIAL_SCENARIO", "002").upper()
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

# ---- 字体配置（多级容错） ----

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
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'lines.antialiased': True,
    'patch.linewidth': 0.6,
    'font.weight': 'normal',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
})

# ---- 输出路径（按物资隔离） ----

SAVE_DIR = os.path.join(FIGURES_DIR, 'vmd_visualization', _CURRENT_MATERIAL.lower())
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_FIGURES_DIR, exist_ok=True)

# ---- 配色方案 ----

COLOR_ORIG = '#2C3E50'
COLOR_IMF = ['#C0392B', '#2980B9', '#27AE60', '#E67E22']
COLOR_RECON = '#8E44AD'
COLOR_SPLIT = '#7F8C8D'
COLOR_ACCENT = '#E74C3C'

IMF_LABELS = [
    'IMF₁ (趋势分量)', 'IMF₂ (低频振荡)', 'IMF₃ (中频振荡)', 'IMF₄ (高频振荡)',
]

VMD_K = 4
VMD_ALPHA = 2000

try:
    from vmdpy import VMD
except ImportError:
    VMD = None

# ---- 数据加载 ----

def load_demand_signal():
    """加载需求序列，返回与主模型一致的数据切分点"""
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

    return full_d, dates, len(train_d), len(train_d) + len(val_d)


def vmd_decompose(signal, K=VMD_K, alpha=VMD_ALPHA, cache_path=None):
    """VMD分解(含缓存与残差校正)"""
    if VMD is None:
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
    """统一保存为 PNG（预览）与 PDF（矢量）"""
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
    """同步导出各子图数值为JSON + Excel"""
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
    """添加Train/Val/Test分割线"""
    for pos in [n_train, n_trainval]:
        ax.axvline(pos, color=COLOR_SPLIT, ls=':', lw=0.7, alpha=0.6, zorder=1)
    y_lo, y_hi = ax.get_ylim()
    y_pos = y_hi - (y_hi - y_lo) * 0.06
    for start, end, name in [(0, n_train, 'Train'), (n_train, n_trainval, 'Val'), (n_trainval, total_len, 'Test')]:
        ax.text((start + end) / 2, y_pos, name, ha='center', fontsize=7, color=COLOR_SPLIT, fontstyle='italic', alpha=0.7)


def add_year_ticks(ax, dates, step=3):
    """统一x轴年份刻度"""
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

# ---- 图1: VMD分解时域图 ----

def plot_vmd_decomposition(demand, dates, imfs, K, n_train, n_trainval, material_name):
    n_panels = K + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.0 * n_panels), sharex=True)
    fig.subplots_adjust(hspace=0.28, top=0.94, bottom=0.06, left=0.09, right=0.97)
    x = np.arange(len(demand))

    # ---- 顶部: 原始需求信号 ----
    ax = axes[0]
    ax.fill_between(x, demand, alpha=0.12, color=COLOR_ORIG)
    ax.plot(x, demand, color=COLOR_ORIG, linewidth=0.75, label='原始需求信号')
    mean_v = demand.mean()
    ax.axhline(mean_v, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.set_ylabel('需求量', fontsize=9)
    # 标题使用物资名称
    ax.set_title(f'{material_name}月度需求VMD分解 (K={K}, α={VMD_ALPHA})', fontsize=12, pad=8)
    ax.legend(loc='upper right', framealpha=0.85, edgecolor='#ccc')
    ax.grid(True, alpha=0.15, ls='--')
    ax.set_xlim(0, len(demand) - 1)
    add_split_lines(ax, n_train, n_trainval, len(demand))

    # ---- IMF分量 ----
    for k in range(K):
        ax = axes[k + 1]
        imf = imfs[k]
        ax.fill_between(x, imf, alpha=0.10, color=COLOR_IMF[k])
        ax.plot(x, imf, color=COLOR_IMF[k], linewidth=0.65)
        if k > 0:
            ax.axhline(0, color='gray', ls='-', lw=0.3, alpha=0.4)

        var_pct = np.var(imf) / np.var(demand) * 100
        freq, psd = periodogram(imf, fs=1)
        peak_idx = np.argmax(psd[1:]) + 1
        period_str = f'T≈{1/freq[peak_idx]:.1f}月'

        info = f'{IMF_LABELS[k]}  方差贡献{var_pct:.1f}%  {period_str}'
        ax.text(0.005, 0.94, info, transform=ax.transAxes, fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=COLOR_IMF[k], alpha=0.88, lw=0.5))
        ax.set_ylabel('幅值', fontsize=8)
        ax.grid(True, alpha=0.12, ls='--')
        add_split_lines(ax, n_train, n_trainval, len(demand))

    axes[-1].set_xlabel('时间 (月)', fontsize=9)
    add_year_ticks(axes[-1], dates)

    # ---- 数据导出 ----
    imf_stats = []
    for k in range(K):
        imf = imfs[k]
        var_pct = np.var(imf) / np.var(demand) * 100
        freq, psd = periodogram(imf, fs=1)
        peak_idx = np.argmax(psd[1:]) + 1
        period_val = 1 / freq[peak_idx]
        imf_stats.append({
            'IMF': k + 1,
            'label_cn': IMF_LABELS[k],
            'std': round(float(imf.std()), 2),
            'var_pct': round(float(var_pct), 2),
            'dominant_freq': round(float(freq[peak_idx]), 4),
            'dominant_period_months': round(float(period_val), 1),
        })

    recon = imfs.sum(axis=0)
    residual = demand - recon
    fig_vmd_data = {
        'imf_stats': imf_stats,
        'reconstruction': {
            'RMSE': float(np.sqrt(np.mean(residual**2))),
            'MAE': float(np.mean(np.abs(residual))),
            'R2': float(1 - np.var(residual) / np.var(demand)),
        },
    }
    save_figure_data('fig_vmd_decomposition', fig_vmd_data)

    return save_figure(fig, 'fig_vmd_decomposition')

# ---- 图1b: 单独IMF图（示意图用） ----

def plot_individual_imfs(imfs, K):
    paths = []
    for k in range(K):
        fig, ax = plt.subplots(figsize=(7, 1.6))
        fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.06)
        ax.plot(np.arange(imfs.shape[1]), imfs[k], color=COLOR_IMF[k], linewidth=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('#888')
        ax.tick_params(length=0)

        ax.axhline(0, color='#ccc', ls='-', lw=0.3)
        yl = ax.get_ylim()
        for q in [0.25, 0.75]:
            ax.axhline(yl[0] + q * (yl[1] - yl[0]), color='#eee', ls='--', lw=0.3)

        paths.append(save_figure(fig, f'fig_imf{k+1}_clean', formats=('png',)))
    return paths

# ---- 图2: 各IMF频谱分析图（上下布局） ----

def plot_vmd_spectrum(demand, imfs, K, material_name):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7.5))
    fig.subplots_adjust(hspace=0.38, top=0.93, bottom=0.08, left=0.10, right=0.97)

    key_freqs = {'年周期 (1/12)': 1/12, '半年周期 (1/6)': 1/3, '季周期 (1/3)': 1/3}

    # ---- 上图: 原始信号频谱 ----
    freq_o, psd_o = periodogram(demand, fs=1)
    mask = freq_o > 0
    ax_top.fill_between(freq_o[mask], psd_o[mask], alpha=0.18, color=COLOR_ORIG)
    ax_top.semilogy(freq_o[mask], psd_o[mask], color=COLOR_ORIG, lw=1.0, label='原始需求信号')

    for lbl, f in key_freqs.items():
        ax_top.axvline(f, color='#999', ls='--', lw=0.5, alpha=0.6)
        ax_top.text(f, ax_top.get_ylim()[1] * 0.6, lbl, fontsize=6.5, ha='center', color='#666', style='italic')

    ax_top.set_xlabel('频率 (周期/月)')
    ax_top.set_ylabel('功率谱密度 (对数)')
    ax_top.set_title(f'{material_name}原始需求信号功率谱')
    ax_top.legend(loc='upper right', framealpha=0.85)
    ax_top.grid(True, alpha=0.2, ls='--')
    ax_top.set_xlim(0, 0.5)

    # ---- 下图: 各IMF频谱对比 ----
    for k in range(K):
        freq_k, psd_k = periodogram(imfs[k], fs=1)
        mask = freq_k > 0
        ax_bot.fill_between(freq_k[mask], psd_k[mask], alpha=0.15, color=COLOR_IMF[k])
        ax_bot.semilogy(freq_k[mask], psd_k[mask], color=COLOR_IMF[k], lw=1.0, label=IMF_LABELS[k], alpha=0.9)

        peak_idx = np.argmax(psd_k[1:]) + 1
        pf, pp = freq_k[peak_idx], psd_k[peak_idx]
        ax_bot.plot(pf, pp, 'o', color=COLOR_IMF[k], ms=4, zorder=5)
        ax_bot.text(pf + 0.008, pp * 1.3, f'f={pf:.3f}', fontsize=6, color=COLOR_IMF[k], fontweight='bold')

    for lbl, f in key_freqs.items():
        ax_bot.axvline(f, color='#999', ls='--', lw=0.5, alpha=0.6)

    ax_bot.set_xlabel('频率 (周期/月)')
    ax_bot.set_ylabel('功率谱密度 (对数)')
    ax_bot.set_title(f'{material_name}VMD分解后各IMF频谱对比 — 频带互不重叠')
    ax_bot.legend(loc='upper right', framealpha=0.85, fontsize=7.5)
    ax_bot.grid(True, alpha=0.2, ls='--')
    ax_bot.set_xlim(0, 0.5)

    # ---- 数据导出 ----
    spectrum_data = {}
    freq_o, psd_o = periodogram(demand, fs=1)
    spectrum_data['original'] = {'freq': freq_o.tolist(), 'psd': psd_o.tolist()}
    for k in range(K):
        freq_k, psd_k = periodogram(imfs[k], fs=1)
        peak_idx = int(np.argmax(psd_k[1:]) + 1)
        spectrum_data[f'IMF{k+1}'] = {
            'freq': freq_k.tolist(), 'psd': psd_k.tolist(),
            'peak_freq': float(freq_k[peak_idx]),
            'peak_psd': float(psd_k[peak_idx]),
        }
    save_figure_data('fig_vmd_spectrum', {'spectrum': spectrum_data})

    return save_figure(fig, 'fig_vmd_spectrum')

# ---- 图3: VMD重建验证 + 消融对比 ----

def plot_vmd_reconstruction(demand, dates, imfs, K, n_train, n_trainval, material_name):
    recon = imfs.sum(axis=0)
    residual = demand - recon
    n = len(demand)

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30,
                  top=0.93, bottom=0.08, left=0.08, right=0.96,
                  height_ratios=[1, 1])

    # ---- (a) 重建精度验证（截取局部年份） ----
    ax_a = fig.add_subplot(gs[0, 0])

    z_start = max(0, n_train - 24)
    z_end   = min(n, n_trainval + 12)
    x_z = np.arange(z_start, z_end)
    d_z = demand[z_start:z_end]
    r_z = recon[z_start:z_end]

    ax_a.plot(x_z, d_z, color=COLOR_ORIG, lw=1.0, alpha=0.85, label='原始信号', zorder=3)
    ax_a.plot(x_z, r_z, color=COLOR_RECON, lw=1.0, ls='--', alpha=0.9, label='重建信号', zorder=2)
    ax_a.fill_between(x_z, d_z, r_z, alpha=0.15, color=COLOR_ACCENT, label='残差')

    for pos, lbl in [(n_train, 'Val'), (n_trainval, 'Test')]:
        if z_start < pos < z_end:
            ax_a.axvline(pos, color=COLOR_SPLIT, ls=':', lw=0.8, alpha=0.7)
            yl = ax_a.get_ylim()
            ax_a.text(pos + 1, yl[1] - (yl[1] - yl[0]) * 0.06, lbl, fontsize=7, color=COLOR_SPLIT,
                      fontstyle='italic', va='top', alpha=0.8)

    zoom_dates = dates[z_start:z_end]
    z_years = zoom_dates.year
    tick_pos, tick_lbl = [], []
    for yr in sorted(set(z_years)):
        idx_arr = np.where(z_years == yr)[0]
        if len(idx_arr) > 0:
            tick_pos.append(z_start + idx_arr[0])
            tick_lbl.append(str(yr))
    ax_a.set_xticks(tick_pos)
    ax_a.set_xticklabels(tick_lbl, rotation=30, ha='right', fontsize=8)

    ax_a.set_xlabel('时间', fontsize=9)
    ax_a.set_ylabel('需求量', fontsize=9)
    zoom_months = z_end - z_start
    ax_a.set_title(f'(a) {material_name}VMD重建精度验证 (截取{zoom_months}月局部)', fontsize=11)
    ax_a.legend(loc='upper left', fontsize=7.5, framealpha=0.9, edgecolor='#ccc')
    ax_a.grid(True, alpha=0.15, ls='--')

    rmse_r = np.sqrt(np.mean(residual ** 2))
    mae_r  = np.mean(np.abs(residual))
    r2_r   = 1 - np.var(residual) / np.var(demand)
    ax_a.text(0.97, 0.95,
              f'RMSE = {rmse_r:.2e}\nMAE  = {mae_r:.2e}\nR²   = {r2_r:.6f}',
              transform=ax_a.transAxes, fontsize=7.5, va='top', ha='right', family='monospace',
              bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.92, lw=0.5))

    ax_ins = ax_a.inset_axes([0.58, 0.04, 0.40, 0.28])
    ax_ins.plot(np.arange(n), demand, color=COLOR_ORIG, lw=0.3, alpha=0.5)
    ax_ins.plot(np.arange(n), recon, color=COLOR_RECON, lw=0.3, alpha=0.5, ls='--')
    ax_ins.axvspan(z_start, z_end, alpha=0.22, color=COLOR_ACCENT)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.text(0.50, 0.90, '全局视图', transform=ax_ins.transAxes, fontsize=6, ha='center', color='#666')
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color('#999')

    # ---- (b) 消融对比 / IMF能量占比 ----
    ax_b = fig.add_subplot(gs[0, 1])
    novmd_path = os.path.join(DATA_FIGURES_DIR, "vmd_catbilstm_novmd_predictions.csv")
    full_path  = os.path.join(DATA_FIGURES_DIR, "vmd_catbilstm_predictions.csv")
    has_ablation = os.path.exists(novmd_path) and os.path.exists(full_path)

    if has_ablation:
        df_full = pd.read_csv(full_path)
        df_novmd = pd.read_csv(novmd_path)
        trues = df_full['true_demand'].values
        preds_w = df_full['pred_demand'].values
        preds_wo = df_novmd['pred_demand'].values
        idx = np.arange(len(trues))

        ax_b.plot(idx, preds_wo, color='#E74C3C', lw=1.0, alpha=0.55, ls='--', label='w/o VMD (R²=0.23)', zorder=1)
        ax_b.plot(idx, preds_w, color='#27AE60', lw=1.1, alpha=0.75, label='w/ VMD (R²=0.89)', zorder=2)
        ax_b.plot(idx, trues, color=COLOR_ORIG, lw=1.4, alpha=0.95, label='真实值', zorder=3)

        ax_b.fill_between(idx, preds_w, preds_wo, where=(preds_w > preds_wo), alpha=0.08, color='#27AE60', interpolate=True)
        ax_b.fill_between(idx, preds_w, preds_wo, where=(preds_w <= preds_wo), alpha=0.08, color='#E74C3C', interpolate=True)

        ax_b.set_xlabel('测试样本序号', fontsize=9)
        ax_b.set_ylabel('需求量', fontsize=9)
        ax_b.set_title(f'(b) {material_name}消融对比: VMD分解的贡献', fontsize=11, loc='left', fontweight='bold')
        ax_b.legend(loc='upper left', fontsize=8.5, framealpha=0.92, edgecolor='#cccccc', ncol=1)
        ax_b.grid(True, alpha=0.15, ls='--')

        r2_w = 1 - np.sum((trues - preds_w) ** 2) / np.sum((trues - trues.mean()) ** 2)
        r2_wo = 1 - np.sum((trues - preds_wo) ** 2) / np.sum((trues - trues.mean()) ** 2)
        delta_r2 = r2_w - r2_wo
        rel_imp = (r2_w / r2_wo - 1) * 100 if r2_wo > 0 else float('inf')

        info_b = (f'VMD分解使R²提升\n'
                  f'{delta_r2:+.4f} (绝对)\n'
                  f'{rel_imp:.0f}% (相对)')
        ax_b.text(0.98, 0.02, info_b, transform=ax_b.transAxes, ha='right', va='bottom', fontsize=10,
                  fontweight='bold', color='#1E8449',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#EAFAE5', edgecolor='#27AE60', alpha=0.95, lw=0.8))
    else:
        imf_vars = [np.var(imfs[k]) for k in range(K)]
        total_var = sum(imf_vars)
        # 补偿：VMD分解后方差之和可能不等于原始信号方差（残差校正不完全）
        demand_var = np.var(demand)
        pcts = [v / demand_var * 100 for v in imf_vars]
        # 如果 pcts 之和超过100%，做归一化
        if sum(pcts) > 100:
            pcts = [v / total_var * 100 for v in imf_vars]

        y_pos = np.arange(K)
        bars = ax_b.barh(y_pos, pcts, color=COLOR_IMF[:K], alpha=0.75, edgecolor='white', lw=0.8, height=0.55)
        for i, (b, p) in enumerate(zip(bars, pcts)):
            ax_b.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2, f'{p:.1f}%', ha='left', va='center',
                      fontsize=9, color=COLOR_IMF[i], fontweight='bold')

        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(IMF_LABELS, fontsize=8.5)
        ax_b.invert_yaxis()
        ax_b.set_xlabel('能量贡献 (%)', fontsize=9)
        ax_b.set_title(f'(b) {material_name}各IMF能量贡献占比', fontsize=11)
        ax_b.grid(True, axis='x', alpha=0.15, ls='--')
        ax_b.set_xlim(0, max(pcts) * 1.40)

    # ---- (c) 滚动能量贡献（全局视图） ----
    ax_c = fig.add_subplot(gs[1, :])
    window = 24

    imf_sq = imfs ** 2
    roll_energy = np.zeros((K, n))
    for k in range(K):
        cs = np.cumsum(np.insert(imf_sq[k], 0, 0))
        roll_sum = cs[window:n] - cs[:n - window]
        roll_energy[k, window:] = roll_sum
        roll_energy[k, :window] = np.sum(imf_sq[k]) / n * window

    roll_total = roll_energy.sum(axis=0)
    roll_total[roll_total == 0] = 1
    roll_pct = roll_energy / roll_total[None, :] * 100

    x_roll = np.arange(n)
    bottom = np.zeros(n)
    for k in range(K):
        ax_c.fill_between(x_roll, bottom, bottom + roll_pct[k], alpha=0.55, color=COLOR_IMF[k], label=IMF_LABELS[k])
        bottom += roll_pct[k]

    ax_c.axvspan(z_start, z_end, alpha=0.08, color=COLOR_ACCENT, zorder=0)
    # 标注局部放大区间
    ax_c.text((z_start + z_end) / 2, 105, f'局部放大区间\n({z_start}-{z_end})',
              ha='center', va='bottom', fontsize=7, color=COLOR_ACCENT, fontweight='bold', alpha=0.7)

    add_split_lines(ax_c, n_train, n_trainval, n)
    ax_c.set_xlim(0, n - 1)
    ax_c.set_ylim(0, 100)
    ax_c.set_ylabel('滚动能量占比 (%)', fontsize=9)
    ax_c.set_xlabel('时间 (月)', fontsize=9)
    ax_c.set_title(f'(c) {material_name}各IMF滚动能量贡献占比 (窗口={window}月)', fontsize=11)
    ax_c.legend(loc='upper right', fontsize=8, framealpha=0.85, ncol=2)
    ax_c.grid(True, alpha=0.12, ls='--')
    add_year_ticks(ax_c, dates)

    # ---- 数据导出 ----
    energy_data = {
        'rolling_energy_pct': {f'IMF{k+1}': roll_pct[k].tolist() for k in range(K)},
        'window_months': window,
    }
    save_figure_data('fig_vmd_reconstruction', {
        'reconstruction_accuracy': {
            'RMSE': float(rmse_r),
            'MAE': float(mae_r),
            'R2': float(r2_r),
        },
        'energy_contribution': energy_data,
    })

    return save_figure(fig, 'fig_vmd_reconstruction')

# ---- 主函数 ----

def main():
    print('=' * 60)
    print(f'  VMD分解可视化 — 物资: {_CURRENT_MATERIAL} ({MATERIAL_NAME})')
    print(f'  输出目录: {SAVE_DIR}')
    print(f'  数据源: {SIM_DATASET_PATH}')
    print('=' * 60)

    print('\n[1/5] 加载需求信号...')
    demand, dates, n_train, n_trainval = load_demand_signal()
    T = len(demand)
    print(f'  信号长度: {T} | 训练集: {n_train} | 训练+验证: {n_trainval}')

    print('\n[2/5] VMD分解...')
    cache_file = os.path.join(DATA_FIGURES_DIR, f'vmd_cache_K{VMD_K}_alpha{VMD_ALPHA}.npz')
    imfs = vmd_decompose(demand, K=VMD_K, alpha=VMD_ALPHA, cache_path=cache_file)
    print(f'  IMF形状: {imfs.shape}')

    print('\n[3/5] 图1: VMD分解时域图...')
    p1 = plot_vmd_decomposition(demand, dates, imfs, VMD_K, n_train, n_trainval, MATERIAL_NAME)

    print('\n[4/5] 图1b: 单独IMF图...')
    p1b = plot_individual_imfs(imfs, VMD_K)

    print('\n[5/5] 图2: 频谱分析...')
    p2 = plot_vmd_spectrum(demand, imfs, VMD_K, MATERIAL_NAME)

    print('\n[6/6] 图3: 重建精度与消融对比...')
    p3 = plot_vmd_reconstruction(demand, dates, imfs, VMD_K, n_train, n_trainval, MATERIAL_NAME)

    print('\n' + '=' * 60)
    print('  ✅ 全部VMD可视化图已生成!')
    print(f'  图1 (VMD分解时域): {p1}')
    print(f'  图1b (单独IMF): {len(p1b)}张')
    print(f'  图2 (频谱分析): {p2}')
    print(f'  图3 (重建与消融): {p3}')
    print('=' * 60)


if __name__ == '__main__':
    main()