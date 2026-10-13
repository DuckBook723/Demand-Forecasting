# -*- coding: utf-8 -*-
"""绘制三种物资的多模型预测曲线对比图（3 行 × 1 列），
数据来自 Forecast_Comparison_Source_Data.csv。"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

warnings.filterwarnings('ignore')

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(os.getcwd())
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入配置（如果存在）
try:
    import configs.config as cfg

    FIGURES_DIR = Path(cfg.FIGURES_DIR)
except ImportError:
    FIGURES_DIR = PROJECT_ROOT / "figures"

# ---- 字体配置 ----
# 尝试加载自定义字体，若失败则使用系统默认
try:
    fm.fontManager.addfont(cfg.FONT_SIMHEI_FILE_PATH)
    fm.fontManager.addfont(cfg.FONT_SARASAMONOSC_REGULAR_FILE_PATH)
    fm.fontManager.addfont(cfg.FONT_SARASAMONOSC_BOLD_FILE_PATH)
    plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'SimHei', 'DejaVu Sans', 'Arial']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']

plt.rcParams['axes.unicode_minus'] = False

# 全局排版参数
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

# ---- 数据与样式配置 ----
MATERIAL_INFO = {
    "001": "10kV油浸式变压器 (001)",
    "002": "架空绝缘导线 (002)",
    "003": "智能电表 (003)",
}
MATERIAL_IDS = ["001", "002", "003"]
SUBFIG_TAGS = ["(a)", "(b)", "(c)"]

# 模型配置，按 CSV 列名映射，列表顺序即绘图顺序
MODEL_REGISTRY_CSV = [
    ("真实值", "True_Demand", "-", 1.8, 0.90, '#2C3E50', False),
    ("本文框架", "Pred_本文框架", "-", 1.5, 0.95, '#D32F2F', True),
    ("VMD-LSTM", "Pred_VMD-LSTM", "--", 0.9, 0.70, '#FF9800', False),
    ("XGBoost", "Pred_XGBoost", "-.", 0.8, 0.60, '#81C784', False),
    ("LSTM", "Pred_LSTM", ":", 0.8, 0.60, '#64B5F6', False),
    ("ARIMA", "Pred_ARIMA", ":", 0.8, 0.60, '#90A4AE', False),
]

SAVE_DIR = FIGURES_DIR / 'forecast_curves_from_csv'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---- CSV 数据读取 ----
def load_data_from_csv(csv_file):
    """从 CSV 读取各物资的真实值与各模型预测值。"""
    df = pd.read_csv(csv_file)
    results = {}

    for mat_id in MATERIAL_IDS:
        mat_df = df[df['Material_ID'] == mat_id].reset_index(drop=True)
        if mat_df.empty:
            continue

        preds = {}
        true_vals = None

        # 提取各模型预测值
        for name, col_name, *_ in MODEL_REGISTRY_CSV:
            if col_name in mat_df.columns:
                vals = mat_df[col_name].values
                if name == "真实值":
                    true_vals = vals
                else:
                    preds[name] = vals

        if true_vals is not None and preds:
            results[mat_id] = {'true': true_vals, 'preds': preds}

    return results


# ---- 主绘图函数 ----
def plot_forecast_curves(csv_file):
    SEP = "=" * 60
    print(SEP)
    print(" 📈 正在生成：异构物资时序预测曲线图（基于CSV数据）...")
    print(SEP)

    # 1. 加载 CSV 数据
    all_data = load_data_from_csv(csv_file)

    if not all_data:
        print("❌ 未在 CSV 中检测到有效数据，请检查文件路径和格式。")
        return

    # 2. 画布构建
    n_mats = len(all_data)
    fig, axes = plt.subplots(
        nrows=n_mats, ncols=1,
        figsize=(12.5, 2.9 * n_mats),
        sharex=False
    )
    if n_mats == 1:
        axes = [axes]

    # 3. 绘图循环
    for idx, mat_id in enumerate(MATERIAL_IDS):
        if mat_id not in all_data:
            continue

        ax = axes[idx]
        data = all_data[mat_id]
        true_vals = data['true']
        preds = data['preds']
        x = np.arange(len(true_vals))

        # ---- (1) 先画基线模型，弱化显示 ----
        for name, col, ls, lw, alpha, color, is_main in MODEL_REGISTRY_CSV:
            if name == "真实值" or is_main:
                continue
            if name in preds:
                ax.plot(x, preds[name], ls=ls, lw=lw, alpha=alpha, color=color, label=name)

        # ---- (2) 再画本文框架，高亮红色 ----
        if "本文框架" in preds:
            ax.plot(x, preds["本文框架"], ls='-', lw=1.5, alpha=0.95, color='#D32F2F', label='本文框架', zorder=5)

        # ---- (3) 最后画真实值，深灰最粗线 ----
        ax.plot(x, true_vals, ls='-', lw=1.8, alpha=0.9, color='#2C3E50', label='真实值', zorder=6)

        # ---- (4) 轴样式调整 ----
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

        # X 轴刻度
        tick_step = max(1, len(x) // 10)
        ax.set_xticks(x[::tick_step])
        ax.set_xlabel('测试集时序步 (时间序列演进)', fontsize=8.2)
        ax.set_xlim(-1, len(x))

        # ---- (6) 局部指标计算与标注 ----
        if "本文框架" in preds:
            ours_pred = preds["本文框架"]
            rmse = np.sqrt(np.mean((true_vals - ours_pred) ** 2))
            mae = np.mean(np.abs(true_vals - ours_pred))
            mask = true_vals != 0
            mape = np.mean(np.abs((true_vals[mask] - ours_pred[mask]) / true_vals[mask])) * 100 if np.any(mask) else 0.0
            r2 = 1 - np.sum((true_vals - ours_pred) ** 2) / np.sum((true_vals - true_vals.mean()) ** 2)

            # 使用 LaTeX 渲染 R²
            metrics_text = rf"$R^2$={r2:.4f} MAE={mae:.1f} MAPE={mape:.2f}%"
            ax.text(
                0.015, 0.95, metrics_text,
                transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#e0e0e0', alpha=0.9, lw=0.6)
            )

    # 总标题
    fig.suptitle(
        f'不同电力物资场景下主流时序预测模型与本文框架预测拟合曲线多维对比（CSV数据驱动）',
        fontsize=12.0, y=0.990, fontweight='bold'
    )

    # 间距紧凑优化
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.06, hspace=0.36)

    # 导出保存
    fig_name = 'fig_forecast_curves_from_csv'
    png_path = SAVE_DIR / f'{fig_name}.png'
    pdf_path = SAVE_DIR / f'{fig_name}.pdf'

    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)

    print(f"\n{SEP}")
    print(f"🎉 【作图完成】预测曲线对比图已生成完毕：")
    print(f" - [超清PNG]: {png_path}")
    print(f" - [学术PDF]: {pdf_path}")
    print(f"{SEP}")


if __name__ == '__main__':
    csv_file = './data/Forecast_Comparison_Source_Data.csv'
    if os.path.exists(csv_file):
        plot_forecast_curves(csv_file)
    else:
        print(f"❌ 文件未找到: {csv_file}")
