# -*- coding: utf-8 -*-
"""
三种物资的多模型预测曲线对比图（3 行 × 1 列，每种物资一行）。

从各物资目录下的 *_predictions.csv 读取数据，文件缺失时用仿真曲线兜底。
真实值用深色粗实线，本文框架用红色高亮，基线模型用半透明细线。
R² 用 LaTeX 的 r'$R^2$' 渲染，输出到 figures/forecast_curves/。
"""

import os
import sys
import json
import warnings
import importlib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

warnings.filterwarnings('ignore')

# ---- 项目路径与配置导入 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent if "__file__" in locals() else Path(os.getcwd())
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import configs.config as cfg

    importlib.reload(cfg)
    FIGURES_DIR = Path(cfg.FIGURES_DIR)
    HAS_CONFIG = True
except Exception as e:
    print(f"⚠️ 导入 configs.config 失败，将采用当前目录作为根目录: {e}")
    FIGURES_DIR = PROJECT_ROOT / "figures"
    HAS_CONFIG = False

# ---- 字体与排版配置 ----
try:
    fm.fontManager.addfont(cfg.FONT_SIMHEI_FILE_PATH)
    fm.fontManager.addfont(cfg.FONT_SARASAMONOSC_REGULAR_FILE_PATH)
    fm.fontManager.addfont(cfg.FONT_SARASAMONOSC_BOLD_FILE_PATH)
    plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'SimHei', 'DejaVu Sans', 'Arial']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']

plt.rcParams['axes.unicode_minus'] = False

# 论文排版参数
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

# ---- 数据配置映射 ----
MATERIAL_INFO = {
    "001": "10kV油浸式变压器 (001)",
    "002": "架空绝缘导线 (002)",
    "003": "智能电表 (003)",
}
MATERIAL_IDS = ["001", "002", "003"]
SUBFIG_TAGS = ["(a)", "(b)", "(c)"]

# (显示名, CSV文件名, 线型, 线宽, 透明度, 颜色, 是否为主模型)
MODEL_REGISTRY = [
    ("真实值", None, "-", 1.8, 0.90, '#2C3E50', False),
    ("本文框架", "vmd_catbilstm_predictions.csv", "-", 1.5, 0.95, '#D32F2F', True),
    ("VMD-LSTM", "vmd_catbilstm_novmd_predictions.csv", "--", 0.9, 0.70, '#FF9800', False),
    ("XGBoost", "xgboost_predictions.csv", "-.", 0.8, 0.60, '#81C784', False),
    ("LSTM", "lstm_diff_predictions.csv", ":", 0.8, 0.60, '#64B5F6', False),
    ("ARIMA", "arima_predictions.csv", ":", 0.8, 0.60, '#90A4AE', False),
]

SAVE_DIR = FIGURES_DIR / 'forecast_curves'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---- 备用仿真数据生成 ----
def generate_mock_predictions(mat_id):
    """本地无 CSV 数据时，生成仿真曲线用于预览"""
    np.random.seed(42 + int(mat_id))
    n_points = 60
    t = np.arange(n_points)

    # 模拟基础时序
    if mat_id == "001":
        base = 500 + 180 * np.sin(t / 4) + 80 * np.cos(t / 8)
        noise_scale = 65
    elif mat_id == "002":
        base = 120 + 40 * np.sin(t / 5) + 15 * np.sin(t / 2)
        noise_scale = 12
    else:
        base = 400 + 90 * np.cos(t / 6) + 30 * np.sin(t / 3)
        noise_scale = 35

    true_vals = np.maximum(base + np.random.normal(0, noise_scale, n_points), 10.0)

    preds = {}
    preds["本文框架"] = np.maximum(base + np.random.normal(0, noise_scale * 0.28, n_points), 10.0)
    preds["VMD-LSTM"] = np.maximum(base + np.random.normal(0, noise_scale * 0.45, n_points) + 8 * np.sin(t / 3), 10.0)
    preds["XGBoost"] = np.maximum(np.roll(base, 1) + np.random.normal(0, noise_scale * 0.55, n_points), 10.0)
    preds["XGBoost"][0] = preds["XGBoost"][1]
    preds["LSTM"] = np.maximum(
        pd.Series(true_vals).rolling(window=3, min_periods=1).mean().values + np.random.normal(0, noise_scale * 0.7,
                                                                                               n_points), 10.0)
    preds["ARIMA"] = np.maximum(pd.Series(true_vals).rolling(window=5, min_periods=1).mean().values + 10, 10.0)

    return true_vals, preds


# ---- 数据解析与对齐 ----
def load_predictions(mat_id):
    """读取指定物资各模型的预测结果"""
    data_dir = FIGURES_DIR / 'data' / mat_id.lower()
    results = {}

    for name, csv_file, *_ in MODEL_REGISTRY:
        if csv_file is None:
            continue

        csv_path = data_dir / csv_file
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
            # 模糊匹配 True/Actual 和 Pred/Forecast 列
            true_col = next((c for c in df.columns if 'true' in c.lower() or 'actual' in c.lower()), None)
            pred_col = next((c for c in df.columns if 'pred' in c.lower() or 'forecast' in c.lower()), None)

            if true_col and pred_col:
                results[name] = {
                    'true': df[true_col].values,
                    'pred': df[pred_col].values,
                }
        except Exception as e:
            print(f"  ⚠️ [{mat_id}] 解析 {csv_file} 失败: {e}")

    if not results:
        return None, None

    # 统一提取真实值（优先取本文框架的真实值）
    if "本文框架" in results:
        true_vals = results["本文框架"]['true']
    else:
        true_vals = next(iter(results.values()))['true']

    # 裁剪对齐长度
    preds = {}
    min_len = len(true_vals)
    for name, data in results.items():
        min_len = min(min_len, len(data['pred']))

    true_vals = true_vals[:min_len]
    for name, data in results.items():
        preds[name] = data['pred'][:min_len]

    return true_vals, preds


# ---- 主绘图 ----
def plot_forecast_curves():
    SEP = "=" * 60
    print(SEP)
    print(" 📈 正在生成：异构物资时序预测曲线图...")
    print(SEP)

    # 1. 尝试加载真实数据
    all_data = {}
    is_real = False

    for mat_id in MATERIAL_IDS:
        print(f"\n📦 物资场景: {MATERIAL_INFO[mat_id]}")
        true_vals, preds = load_predictions(mat_id)

        if true_vals is not None and len(true_vals) > 0:
            all_data[mat_id] = {'true': true_vals, 'preds': preds}
            is_real = True
            print(f"   ✅ [真实读取成功] 数据点总长度={len(true_vals)}, 成功对齐模型数量={len(preds)}")
        else:
            # 回退：生成仿真备份数据
            true_vals, preds = generate_mock_predictions(mat_id)
            all_data[mat_id] = {'true': true_vals, 'preds': preds}
            print(f"   ⚠️  [未检测到CSV] 启用备份仿真数据渲染，保障制图完整性。")

    # 2. 画布构建
    n_mats = len(all_data)
    fig, axes = plt.subplots(
        nrows=n_mats, ncols=1,
        figsize=(12.5, 2.9 * n_mats),
        sharex=False
    )
    if n_mats == 1:
        axes = [axes]

    json_export = {}

    for idx, mat_id in enumerate(MATERIAL_IDS):
        ax = axes[idx]
        data = all_data[mat_id]
        true_vals = data['true']
        preds = data['preds']
        x = np.arange(len(true_vals))

        # ---- (1) 先绘制基线模型 (弱化显示) ----
        for name, csv_file, ls, lw, alpha, color, is_main in MODEL_REGISTRY:
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

        # X轴刻度设置
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

            # 用 LaTeX 的 r"$R^2$" 渲染决定系数，避免符号在部分终端乱码
            metrics_text = rf"$R^2$={r2:.4f}   MAE={mae:.1f}   MAPE={mape:.2f}%"
            ax.text(
                0.015, 0.95, metrics_text,
                transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#e0e0e0', alpha=0.9, lw=0.6)
            )

        # JSON 记录
        json_export[mat_id] = {
            'true': true_vals.tolist(),
            'models': {k: v.tolist() for k, v in preds.items()},
        }

    # 3. 全局总标题
    title_suffix = "（真实运行数据）" if is_real else "（全量演示仿真曲线）"
    fig.suptitle(
        f'不同电力物资场景下主流时序预测模型与本文框架预测拟合曲线多维对比{title_suffix}',
        fontsize=12.0, y=0.990, fontweight='bold'
    )

    # 间距紧凑优化
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.06, hspace=0.36)

    # 4. 文件导出保存
    fig_name = 'fig_forecast_curves_comparison'
    png_path = SAVE_DIR / f'{fig_name}.png'
    pdf_path = SAVE_DIR / f'{fig_name}.pdf'
    json_path = SAVE_DIR / f'{fig_name}_data.json'

    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_export, f, ensure_ascii=False, indent=2)

    print(f"\n{SEP}")
    print(f"🎉  【作图完成】预测曲线对比图已经全部生成完毕：")
    print(f"    - [超清PNG]: {png_path}")
    print(f"    - [学术PDF]: {pdf_path}")
    print(f"    - [数据JSON]: {json_path}")
    print(f"{SEP}")


if __name__ == '__main__':
    plot_forecast_curves()
