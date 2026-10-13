# -*- coding: utf-8 -*-
"""
异构物资需求预测性能对比图（3 行 × 3 列，每行一个物资：绝对误差 / 相对误差 / 拟合优度）。

指标从 figures/data/<物资>/ 下的各模型结果 CSV 读取，读不到时用内置兜底数据。
R² 用 LaTeX 的 r'$R^2$' 渲染，输出到 figures/forecast_performance/。
"""

import os
import sys
import json
import warnings
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

# ---- 全局排版配置 ----
plt.rcParams.update({
    'font.size': 8.0,
    'axes.titlesize': 9.5,
    'axes.labelsize': 8.0,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.75,
})
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ---- 实验配置与映射 ----
MODELS = ["ARIMA", "XGBoost", "LSTM", "LSTM-diff", "本文框架 (Ours)"]

MODEL_FILES = {
    "ARIMA": "arima_results.csv",
    "XGBoost": "xgboost_results.csv",
    "LSTM": "fair_lstm_bilstm_results.csv",
    "LSTM-diff": "lstm_diff_results_seed42.csv",
    "本文框架 (Ours)": "vmd_catbilstm_fs_results.csv"
}

MATERIAL_IDS = ["001", "002", "003"]
MATERIAL_NAMES = {
    "001": "10kV油浸式变压器 (001)",
    "002": "架空绝缘导线 (002)",
    "003": "智能电表 (003)"
}

# 兜底数据（含 R²）
MOCK_METRICS = {
    "001": {
        "MAE": [408.1, 326.5, 569.3, 386.6, 246.1],
        "RMSE": [608.3, 556.8, 773.7, 587.6, 346.6],
        "MAPE": [9.16, 7.16, 12.77, 8.60, 5.74],
        "R2": [0.5405, 0.6149, 0.3231, 0.5691, 0.8508]
    },
    "002": {
        "MAE": [75.5, 55.7, 108.1, 70.9, 71.4],
        "RMSE": [97.6, 69.0, 131.7, 86.7, 88.5],
        "MAPE": [4.15, 3.13, 5.95, 3.91, 3.98],
        "R2": [0.8392, 0.9196, 0.7499, 0.8749, 0.8678]
    },
    "003": {
        "MAE": [392.3, 357.9, 482.5, 390.5, 259.6],
        "RMSE": [501.3, 459.7, 580.4, 498.6, 326.3],
        "MAPE": [7.92, 7.27, 10.06, 8.01, 5.65],
        "R2": [0.8415, 0.8667, 0.7940, 0.8431, 0.9328]
    }
}

COLOR_MAP = {
    "ARIMA": "#90A4AE",
    "XGBoost": "#81C784",
    "LSTM": "#64B5F6",
    "LSTM-diff": "#FFB74D",
    "本文框架 (Ours)": "#D32F2F"
}


# ---- 数据解析工具 ----
def extract_metrics_from_row(row_dict):
    """提取行字典中的指标，处理大小写与上标变体"""
    lower_map = {str(k).lower().replace('\u00b2', '2').strip(): v for k, v in row_dict.items()}

    def get_val(*keys):
        for k in keys:
            if k.lower() in lower_map:
                v = lower_map[k.lower()]
                if pd.notna(v) and v not in (None, "", "nan"):
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        continue
        return np.nan

    return {
        "RMSE": get_val("RMSE", "rmse"),
        "MAE": get_val("MAE", "mae"),
        "MAPE": get_val("MAPE", "mape"),
        "R2": get_val("R2", "r2", "R^2", "r^2"),
    }


def load_real_results():
    """遍历物资目录，解析各模型的结果 CSV"""
    real_data = {}
    found_any_file = False

    print("\n" + "=" * 60)
    print("🔄 开始从项目目录结构扫描实验结果并提取多维指标 (含 R²)...")
    print(f"📂 目标数据目录: {FIGURES_DIR / 'data'}")

    for m_id in MATERIAL_IDS:
        material_dir = FIGURES_DIR / "data" / m_id.lower()
        metrics_for_mat = {"MAE": [], "RMSE": [], "MAPE": [], "R2": []}

        print(f"\n📦 物资场景: {MATERIAL_NAMES[m_id]}")

        for model_name in MODELS:
            csv_name = MODEL_FILES[model_name]
            csv_path = material_dir / csv_name

            mae_val, rmse_val, mape_val, r2_val = np.nan, np.nan, np.nan, np.nan

            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        df.columns = [c.strip().replace('\u00b2', '2') for c in df.columns]
                        col_lower = {c.lower(): c for c in df.columns}
                        r2_col = col_lower.get('r2', None)
                        if r2_col and len(df) > 1:
                            df = df.sort_values(r2_col, ascending=False).head(1)

                        row = df.iloc[0].to_dict()
                        extracted = extract_metrics_from_row(row)

                        mae_val = extracted["MAE"]
                        rmse_val = extracted["RMSE"]
                        mape_val = extracted["MAPE"]
                        r2_val = extracted["R2"]
                        found_any_file = True
                        print(
                            f"   ✅ [已读取] {model_name:14s} -> MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MAPE={mape_val:.2f}%, R²={r2_val:.4f}")
                except Exception as e:
                    print(f"   ⚠️  [读取出错] {model_name:14s} (文件: {csv_name}): {e}")
            else:
                print(f"   ❌ [未跑完/缺失] {model_name:14s} (期望路径: {m_id.lower()}/{csv_name})")

            metrics_for_mat["MAE"].append(mae_val)
            metrics_for_mat["RMSE"].append(rmse_val)
            metrics_for_mat["MAPE"].append(mape_val)
            metrics_for_mat["R2"].append(r2_val)

        real_data[m_id] = {
            "name": MATERIAL_NAMES[m_id],
            "metrics": metrics_for_mat
        }

    return real_data, found_any_file


# ---- 绘图与可视化 ----
def plot_and_save():
    # 1. 加载数据
    data_dict, is_real = load_real_results()

    if not is_real:
        print("\n📢 [系统提示] 未在项目 outputs 目录中扫描到任何真实实验结果。")
        print("💡 [降级启用] 脚本已为您加载模拟学术备份数据！")
        for m_id in MATERIAL_IDS:
            data_dict[m_id] = {
                "name": MATERIAL_NAMES[m_id],
                "metrics": MOCK_METRICS[m_id]
            }

    # 2. 画布构建
    fig, axes = plt.subplots(
        nrows=3, ncols=3,
        figsize=(15.5, 9.5),
        gridspec_kw={
            'width_ratios': [1.2, 0.8, 0.8],  # 列宽控制
            'hspace': 0.30,
            'wspace': 0.16
        }
    )

    y_pos = np.arange(len(MODELS))
    bar_height = 0.35

    for idx, mat_id in enumerate(MATERIAL_IDS):
        mat_data = data_dict[mat_id]
        mat_name = mat_data["name"]
        metrics = mat_data["metrics"]

        # ---- 左列：MAE / RMSE 绝对误差 ----
        ax_left = axes[idx, 0]
        model_colors = [COLOR_MAP[m] for m in MODELS]

        mae_clean = [0.0 if np.isnan(v) else v for v in metrics["MAE"]]
        rmse_clean = [0.0 if np.isnan(v) else v for v in metrics["RMSE"]]

        mae_bars = ax_left.barh(y_pos + bar_height / 2, mae_clean, height=bar_height, color=model_colors, alpha=0.9)
        rmse_bars = ax_left.barh(y_pos - bar_height / 2, rmse_clean, height=bar_height, color=model_colors,
                                 edgecolor='white', linewidth=0.5, hatch='///', alpha=0.6)

        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(MODELS, fontweight='bold')
        ax_left.invert_yaxis()
        ax_left.grid(True, axis='x', linestyle='--', alpha=0.15)
        ax_left.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_left.spines['top'].set_visible(False)
        ax_left.spines['right'].set_visible(False)
        ax_left.set_title(f"({chr(97 + idx * 3)}) {mat_name} - 绝对误差 (MAE/RMSE)", pad=8, fontweight='bold')
        ax_left.set_xlabel("误差值 (实物物理单位)", fontsize=7.5)

        # 标注文本
        max_val_left = max(max(mae_clean), max(rmse_clean)) if max(mae_clean) > 0 else 100.0
        for i, bar in enumerate(mae_bars):
            val = metrics["MAE"][i]
            val_text = f'{val:.1f}' if pd.notna(val) else "N/A"
            ax_left.text(bar.get_width() + (max_val_left * 0.012), bar.get_y() + bar.get_height() / 2, val_text,
                         va='center', ha='left', fontsize=6.5, color='#333333')
        for i, bar in enumerate(rmse_bars):
            val = metrics["RMSE"][i]
            val_text = f'{val:.1f}' if pd.notna(val) else "N/A"
            ax_left.text(bar.get_width() + (max_val_left * 0.012), bar.get_y() + bar.get_height() / 2, val_text,
                         va='center', ha='left', fontsize=6.5, color='#666666',
                         fontstyle='italic' if pd.notna(val) else 'normal')

        # 第一行添加图例
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#555555', alpha=0.9, label='MAE (平均绝对误差)'),
                Patch(facecolor='#555555', alpha=0.6, hatch='///', edgecolor='white', label='RMSE (均方根误差)'),
            ]
            ax_left.legend(handles=legend_elements, loc='lower right', frameon=True, edgecolor='#e0e0e0',
                           facecolor='white', framealpha=0.9)

        # ---- 中列：MAPE 相对误差 ----
        ax_mid = axes[idx, 1]
        mape_clean = [0.0 if np.isnan(v) else v for v in metrics["MAPE"]]
        mape_bars = ax_mid.barh(y_pos, mape_clean, height=bar_height * 1.5, color=model_colors, alpha=0.85)

        ax_mid.set_yticks(y_pos)
        ax_mid.set_yticklabels([])
        ax_mid.invert_yaxis()
        ax_mid.grid(True, axis='x', linestyle='--', alpha=0.15)
        ax_mid.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_mid.spines['top'].set_visible(False)
        ax_mid.spines['right'].set_visible(False)
        ax_mid.set_title(f"({chr(97 + idx * 3 + 1)}) {mat_name} - 相对误差 (MAPE)", pad=8, fontweight='bold')
        ax_mid.set_xlabel("MAPE (%)", fontsize=7.5)

        max_val_mid = max(mape_clean) if max(mape_clean) > 0 else 10.0
        for i, bar in enumerate(mape_bars):
            val = metrics["MAPE"][i]
            val_text = f'{val:.2f}%' if pd.notna(val) else "N/A"
            is_ours = (bar.get_facecolor()[:3] == matplotlib.colors.to_rgb('#D32F2F'))
            ax_mid.text(
                bar.get_width() + (max_val_mid * 0.015),
                bar.get_y() + bar.get_height() / 2,
                val_text,
                va='center', ha='left',
                fontsize=6.8,
                fontweight='bold' if (is_ours and pd.notna(val)) else 'normal'
            )

        # ---- 右列：R² 拟合优度 ----
        ax_right = axes[idx, 2]
        r2_clean = [0.0 if (np.isnan(v) or v < 0) else v for v in metrics["R2"]]
        r2_bars = ax_right.barh(y_pos, r2_clean, height=bar_height * 1.5, color=model_colors, alpha=0.85)

        ax_right.set_yticks(y_pos)
        ax_right.set_yticklabels([])
        ax_right.invert_yaxis()
        ax_right.grid(True, axis='x', linestyle='--', alpha=0.15)

        ax_right.set_xlim(0, 1.15)
        ax_right.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)

        # 用 LaTeX 的 r'$R^2$' 渲染，避免上标显示异常
        ax_right.set_title(f"({chr(97 + idx * 3 + 2)}) {mat_name} - 拟合优度 ($R^2$)", pad=8, fontweight='bold')
        ax_right.set_xlabel(r"$R^2$ 决定系数 (逼近 1.0 为最优)", fontsize=7.5)

        for i, bar in enumerate(r2_bars):
            val = metrics["R2"][i]
            val_text = f'{val:.4f}' if pd.notna(val) else "N/A"
            is_ours = (bar.get_facecolor()[:3] == matplotlib.colors.to_rgb('#D32F2F'))
            ax_right.text(
                bar.get_width() + 0.015,
                bar.get_y() + bar.get_height() / 2,
                val_text,
                va='center', ha='left',
                fontsize=6.8,
                fontweight='bold' if (is_ours and pd.notna(val)) else 'normal'
            )

    # 4. 全局总标题 (同样采用 LaTeX r'$R^2$' 渲染)
    title_suffix = "（真实运行数据）" if is_real else "（全量演示仿真结果）"
    fig.suptitle(
        r"不同形态电力物资场景下本文预测框架与基准模型多维性能（MAE/RMSE/MAPE/$R^2$）对比" + title_suffix,
        fontsize=12.0, y=0.988, fontweight='bold'
    )

    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.06)

    # 5. 保存输出
    save_dir = FIGURES_DIR / "forecast_performance"
    save_dir.mkdir(parents=True, exist_ok=True)

    png_path = save_dir / "fig_forecast_performance_comparison.png"
    pdf_path = save_dir / "fig_forecast_performance_comparison.pdf"

    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)

    print(f"\n📊 【作图完成】3×3全维度对比图已成功保存至: \n   - [PNG]: {png_path} \n   - [PDF]: {pdf_path}")


# ---- 脚本入口 ----
if __name__ == '__main__':
    plot_and_save()
