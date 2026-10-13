# -*- coding: utf-8 -*-
"""
基于 json_to_csv_converter.py 导出的 CSV 源数据重绘预测曲线对比图。

排版配色与 plot_forecast_curves.py 保持一致：3 行 × 1 列，真实值深色粗线 +
本文框架红色高亮 + 基线模型半透明细线，右上角标注 R²/MAE/MAPE。

用法（项目根目录执行）：
  python src/visualization/plot_forecast_curves_from_csv.py
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

warnings.filterwarnings('ignore')

# ---- 项目根路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import configs.config as cfg
    FIGURES_DIR = Path(cfg.FIGURES_DIR)
except Exception:
    FIGURES_DIR = PROJECT_ROOT / "figures"

# ---- 字体与排版配置（与原始文件一致） ----
try:
    fm.fontManager.addfont(cfg.FONT_SIMHEI_FILE_PATH)
    fm.fontManager.addfont(cfg.FONT_SARASAMONOSC_REGULAR_FILE_PATH)
    fm.fontManager.addfont(cfg.FONT_SARASAMONOSC_BOLD_FILE_PATH)
    plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'SimHei', 'DejaVu Sans', 'Arial']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']

plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({
    'font.size': 8.5,
    'axes.titlesize': 10.0,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.75,
})

# ---- 配置映射（与原始文件一致） ----
MATERIAL_INFO = {
    "001": "10kV油浸式变压器 (001)",
    "002": "架空绝缘导线 (002)",
    "003": "智能电表 (003)",
}
MATERIAL_IDS = ["001", "002", "003"]
SUBFIG_TAGS = ["(a)", "(b)", "(c)"]

MODEL_REGISTRY = [
    ("真实值",    "-",  1.8, 0.90, '#2C3E50', False),
    ("本文框架",  "-",  1.5, 0.95, '#D32F2F', True),
    ("VMD-LSTM", "--", 0.9, 0.70, '#FF9800', False),
    ("XGBoost",  "-.", 0.8, 0.60, '#81C784', False),
    ("LSTM",     ":",  0.8, 0.60, '#64B5F6', False),
    ("ARIMA",    ":",  0.8, 0.60, '#90A4AE', False),
]

SAVE_DIR = FIGURES_DIR / 'forecast_curves_from_csv'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_data_from_csv():
    """从 CSV 读取数据，转换为与原始 load_predictions() 相同的返回格式"""
    csv_path = SAVE_DIR / 'forecast_curves_data_from_json.csv'
    if not csv_path.exists():
        print(f"[ERROR] 未找到 CSV 文件: {csv_path}")
        print("请先执行: python src/visualization/json_to_csv_converter.py")
        return None

    df = pd.read_csv(csv_path, dtype={'material_id': str})

    all_data = {}
    for mat_id in MATERIAL_IDS:
        mat_df = df[df['material_id'] == mat_id]

        # 提取真实值
        true_df = mat_df[mat_df['model'] == '真实值'].sort_values('time_step')
        true_vals = true_df['value'].values

        # 提取各模型预测值
        pred_model_names = ["本文框架", "VMD-LSTM", "XGBoost", "LSTM", "ARIMA"]
        preds = {}
        for name in pred_model_names:
            model_df = mat_df[mat_df['model'] == name].sort_values('time_step')
            if len(model_df) > 0:
                preds[name] = model_df['value'].values

        # 对齐长度（与原始逻辑一致）
        min_len = len(true_vals)
        for v in preds.values():
            min_len = min(min_len, len(v))

        true_vals = true_vals[:min_len]
        preds = {k: v[:min_len] for k, v in preds.items()}

        all_data[mat_id] = {'true': true_vals, 'preds': preds}

    return all_data


def plot_forecast_curves():
    print("=" * 60)
    print(" 基于 CSV 源数据的预测曲线对比图")
    print("=" * 60)

    # 1. 加载数据
    all_data = load_data_from_csv()
    if all_data is None:
        return

    # 2. 画布构建（与原始文件一致）
    n_mats = len(MATERIAL_IDS)
    fig, axes = plt.subplots(
        nrows=n_mats, ncols=1,
        figsize=(12.5, 2.9 * n_mats),
        sharex=False
    )
    if n_mats == 1:
        axes = [axes]

    for idx, mat_id in enumerate(MATERIAL_IDS):
        ax = axes[idx]
        data = all_data[mat_id]
        true_vals = data['true']
        preds = data['preds']
        x = np.arange(len(true_vals))

        # ---- (1) 先绘制基线模型 (弱化显示) ----
        for name, ls, lw, alpha, color, is_main in MODEL_REGISTRY:
            if name == "真实值" or is_main:
                continue
            if name not in preds:
                continue
            ax.plot(x, preds[name], ls=ls, lw=lw, alpha=alpha, color=color, label=name)

        # ---- (2) 再绘制本文框架（红色高亮） ----
        if "本文框架" in preds:
            ax.plot(x, preds["本文框架"], ls='-', lw=1.5, alpha=0.95, color='#D32F2F', label='本文框架', zorder=5)

        # ---- (3) 最后绘制真实值（深灰粗线，置顶） ----
        ax.plot(x, true_vals, ls='-', lw=1.8, alpha=0.9, color='#2C3E50', label='真实值', zorder=6)

        # ---- (4) 轴样式 ----
        ax.set_title(f'{SUBFIG_TAGS[idx]} {MATERIAL_INFO[mat_id]}', pad=6, fontweight='bold')
        ax.set_ylabel('物资需求量 (实物单位)', fontsize=8.2)
        ax.grid(True, linestyle='--', alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ---- (5) 图例重排序 ----
        handles, labels = ax.get_legend_handles_labels()
        order = ['真实值', '本文框架']
        ordered_handles = []
        ordered_labels = []
        for item in order:
            if item in labels:
                i = labels.index(item)
                ordered_handles.append(handles[i])
                ordered_labels.append(labels[i])
        for h, l in zip(handles, labels):
            if l not in order:
                ordered_handles.append(h)
                ordered_labels.append(l)

        ax.legend(
            ordered_handles, ordered_labels,
            loc='upper right', framealpha=0.88, edgecolor='#e0e0e0',
            facecolor='white', ncol=3, fontsize=7.2
        )

        # ---- (6) X轴刻度 ----
        tick_step = max(1, len(x) // 10)
        ax.set_xticks(x[::tick_step])
        ax.set_xlabel('测试集时序步 (时间序列演进)', fontsize=8.2)
        ax.set_xlim(-1, len(x))

        # ---- (7) 局部指标计算与标注 ----
        if "本文框架" in preds:
            ours_pred = preds["本文框架"]
            rmse = np.sqrt(np.mean((true_vals - ours_pred) ** 2))
            mae = np.mean(np.abs(true_vals - ours_pred))
            mask = true_vals != 0
            mape = np.mean(np.abs((true_vals[mask] - ours_pred[mask]) / true_vals[mask])) * 100 if np.any(mask) else 0.0
            r2 = 1 - np.sum((true_vals - ours_pred) ** 2) / np.sum((true_vals - true_vals.mean()) ** 2)

            metrics_text = rf"$R^2$={r2:.4f}   MAE={mae:.1f}   MAPE={mape:.2f}%"
            ax.text(
                0.015, 0.95, metrics_text,
                transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#e0e0e0', alpha=0.9, lw=0.6)
            )

    # 3. 全局总标题
    fig.suptitle(
        '不同电力物资场景下主流时序预测模型与本文框架预测拟合曲线多维对比（基于CSV源数据）',
        fontsize=12.0, y=0.990, fontweight='bold'
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.06, hspace=0.36)

    # 4. 文件保存
    fig_name = 'fig_forecast_curves_from_csv'
    png_path = SAVE_DIR / f'{fig_name}.png'
    pdf_path = SAVE_DIR / f'{fig_name}.pdf'

    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)

    print(f"\n  [OK] 图像已保存:")
    print(f"    - PNG: {png_path}")
    print(f"    - PDF: {pdf_path}")


if __name__ == '__main__':
    plot_forecast_curves()