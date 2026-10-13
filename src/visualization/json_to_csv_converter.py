# -*- coding: utf-8 -*-
"""
把绘图时导出的 JSON 源数据转成 CSV，方便直接查看和复用。

输入：
  figures/forecast_curves/fig_forecast_curves_comparison_data.json
  figures/vmd_visualization/combined_final/fig_vmd_combined_multi_material_final_data.json

输出（与原 JSON 同目录）：
  forecast_curves_data_from_json.csv    各模型预测值与真实值
  vmd_signal_data_from_json.csv         全序列信号与各模态分量
  vmd_imf_stats_from_json.csv           各模态统计特征

用法（项目根目录执行）：
  python src/visualization/json_to_csv_converter.py
"""

import os
import sys
import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def convert_forecast_json_to_csv():
    """
    读取预测曲线对比图的 JSON，输出长格式 CSV（物资、模型、时间步、数值）。
    """
    json_path = PROJECT_ROOT / 'figures' / 'forecast_curves' / 'fig_forecast_curves_comparison_data.json'
    if not json_path.exists():
        print(f"[WARN] 未找到文件: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    material_names = {
        "001": "10kV油浸式变压器",
        "002": "架空绝缘导线",
        "003": "智能电表",
    }

    all_rows = []

    for mat_id in ["001", "002", "003"]:
        if mat_id not in data:
            continue

        mat_data = data[mat_id]
        true_vals = mat_data.get('true', [])
        models_dict = mat_data.get('models', {})

        # 真实值
        for i, v in enumerate(true_vals):
            all_rows.append({
                'material_id': mat_id,
                'material_name': material_names[mat_id],
                'model': '真实值',
                'time_step': i,
                'value': v,
            })

        # 各模型预测值
        for model_name, pred_vals in models_dict.items():
            for i, v in enumerate(pred_vals):
                if i >= len(true_vals):
                    break
                all_rows.append({
                    'material_id': mat_id,
                    'material_name': material_names[mat_id],
                    'model': model_name,
                    'time_step': i,
                    'value': v,
                })

    if not all_rows:
        print("[WARN] 未读取到任何预测曲线数据")
        return None

    df = pd.DataFrame(all_rows)
    # 确保按物资、模型、时间步排序
    df['model_order'] = df['model'].map({
        '真实值': 0, '本文框架': 1, 'VMD-LSTM': 2,
        'XGBoost': 3, 'LSTM': 4, 'ARIMA': 5,
    }).fillna(99)
    df = df.sort_values(['material_id', 'model_order', 'time_step']).drop(columns=['model_order']).reset_index(drop=True)

    out_path = PROJECT_ROOT / 'figures' / 'forecast_curves' / 'forecast_curves_data_from_json.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 预测曲线数据已导出: {out_path}")
    print(f"     数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"     包含物资: {df['material_id'].unique().tolist()}")
    print(f"     包含模型: {df['model'].unique().tolist()}")
    return out_path


def convert_vmd_json_to_csv():
    """
    读取 VMD 分解图的 JSON，输出两个 CSV：全序列信号表和各 IMF 统计特征表。
    """
    json_path = PROJECT_ROOT / 'figures' / 'vmd_visualization' / 'combined_final' / 'fig_vmd_combined_multi_material_final_data.json'
    if not json_path.exists():
        print(f"[WARN] 未找到文件: {json_path}")
        return None, None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    material_ids = data.get('figure_meta', {}).get('materials', ["001", "002", "003"])

    # ---- 导出信号表 ----
    signal_rows = []
    for mat_id in material_ids:
        signal_table = data.get(f'{mat_id}_signal_table', [])
        for row in signal_table:
            signal_rows.append({
                'material_id': mat_id,
                'material_name': data.get(f'{mat_id}_meta', {}).get('material_name', ''),
                'date': row.get('date', ''),
                'demand': row.get('demand', ''),
                'IMF1': row.get('IMF1', ''),
                'IMF2': row.get('IMF2', ''),
                'IMF3': row.get('IMF3', ''),
                'IMF4': row.get('IMF4', ''),
            })

    signal_out_path = None
    if signal_rows:
        df_signal = pd.DataFrame(signal_rows)
        # 按物资、日期排序
        df_signal['date_dt'] = pd.to_datetime(df_signal['date'])
        df_signal = df_signal.sort_values(['material_id', 'date_dt']).drop(columns=['date_dt']).reset_index(drop=True)
        signal_out_path = PROJECT_ROOT / 'figures' / 'vmd_visualization' / 'combined_final' / 'vmd_signal_data_from_json.csv'
        df_signal.to_csv(signal_out_path, index=False, encoding='utf-8-sig')
        print(f"[OK] VMD信号数据已导出: {signal_out_path}")
        print(f"     数据维度: {df_signal.shape[0]} 行 x {df_signal.shape[1]} 列")
    else:
        print("[WARN] 未读取到VMD信号数据")

    # ---- 导出统计特征表 ----
    stats_rows = []
    for mat_id in material_ids:
        imf_stats = data.get(f'{mat_id}_imf_stats', [])
        recon = data.get(f'{mat_id}_reconstruction', {})
        for imf_row in imf_stats:
            stats_rows.append({
                'material_id': mat_id,
                'material_name': data.get(f'{mat_id}_meta', {}).get('material_name', ''),
                'IMF': imf_row.get('imf', ''),
                'label': imf_row.get('label', ''),
                'mean': imf_row.get('mean', ''),
                'std': imf_row.get('std', ''),
                'var_pct': imf_row.get('var_pct', ''),
                'dominant_freq': imf_row.get('dominant_freq', ''),
                'dominant_period_months': imf_row.get('dominant_period_months', ''),
                'reconstruction_R2': recon.get('R2', ''),
                'reconstruction_RMSE': recon.get('RMSE', ''),
                'reconstruction_MAE': recon.get('MAE', ''),
            })

    stats_out_path = None
    if stats_rows:
        df_stats = pd.DataFrame(stats_rows)
        stats_out_path = PROJECT_ROOT / 'figures' / 'vmd_visualization' / 'combined_final' / 'vmd_imf_stats_from_json.csv'
        df_stats.to_csv(stats_out_path, index=False, encoding='utf-8-sig')
        print(f"[OK] VMD统计特征数据已导出: {stats_out_path}")
        print(f"     数据维度: {df_stats.shape[0]} 行 x {df_stats.shape[1]} 列")
    else:
        print("[WARN] 未读取到VMD统计特征数据")

    return signal_out_path, stats_out_path


def main():
    print("=" * 70)
    print(" JSON 源数据 → CSV 转换工具")
    print(" 从绘图导出的 JSON 文件中提取源数据并保存为 CSV")
    print("=" * 70)

    print("\n[1/2] 处理预测曲线对比图 JSON ...")
    convert_forecast_json_to_csv()

    print("\n[2/2] 处理VMD分解结果图 JSON ...")
    convert_vmd_json_to_csv()

    print("\n" + "=" * 70)
    print(" 转换完成！CSV 文件已与原 JSON 文件保存在同一目录下。")
    print("=" * 70)


if __name__ == '__main__':
    main()