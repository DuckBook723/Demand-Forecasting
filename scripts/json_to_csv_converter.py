import json
import pandas as pd
import os


def convert_forecast_json(json_path, output_csv):
    """将预测对比 JSON 转换为 CSV，列为 Material_ID、Time_Index、True_Demand
    以及各模型的 Pred_* 预测列。"""
    print(f"正在处理: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_data = []

    # 遍历三种物资
    for mat_id, content in data.items():
        true_vals = content['true']
        models_preds = content['models']

        # 构建基础字典
        base_dict = {
            'Material_ID': mat_id,
            'Time_Index': list(range(len(true_vals))),
            'True_Demand': true_vals
        }

        # 添加各模型预测值 (宽表格式，便于对比)
        for model_name, preds in models_preds.items():
            # 列名标准化：Pred_本文框架, Pred_ARIMA 等
            col_name = f'Pred_{model_name}'
            # 确保长度对齐
            if len(preds) == len(true_vals):
                base_dict[col_name] = preds
            else:
                print(f"  ⚠️ [{mat_id}] {model_name} 长度不一致，已截断对齐")
                min_len = min(len(preds), len(true_vals))
                base_dict[col_name] = preds[:min_len]
                # 同时截断其他列
                for key in ['Time_Index', 'True_Demand']:
                    base_dict[key] = base_dict[key][:min_len]

        # 转换为 DataFrame
        df = pd.DataFrame(base_dict)
        all_data.append(df)

    # 合并所有物资数据
    final_df = pd.concat(all_data, ignore_index=True)

    # 保存 CSV
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"  ✅ 已导出: {output_csv} (共 {len(final_df)} 行)")


def convert_vmd_json(json_path, output_csv_signal, output_csv_stats):
    """将 VMD 分解 JSON 拆成两个 CSV：信号数据用于绘图，IMF 统计信息用于分析。"""
    print(f"正在处理: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取绘图数据
    signal_dfs = []
    for mat_id in ['001', '002', '003']:
        key = f'{mat_id}_signal_table'
        if key in data:
            df_sig = pd.DataFrame(data[key])
            df_sig['Material_ID'] = mat_id

            # 重命名列名，使其更具可读性
            rename_map = {
                'demand': 'Original_Demand',
                'IMF1': 'IMF1_Trend',
                'IMF2': 'IMF2_LowFreq',
                'IMF3': 'IMF3_MidFreq',
                'IMF4': 'IMF4_HighFreq'
            }
            df_sig = df_sig.rename(columns=rename_map)

            # 调整列顺序
            cols_order = ['Material_ID', 'date', 'Original_Demand',
                          'IMF1_Trend', 'IMF2_LowFreq', 'IMF3_MidFreq', 'IMF4_HighFreq']
            df_sig = df_sig[cols_order]

            signal_dfs.append(df_sig)

    if signal_dfs:
        final_signal_df = pd.concat(signal_dfs, ignore_index=True)
        final_signal_df.to_csv(output_csv_signal, index=False, encoding='utf-8-sig')
        print(f"  ✅ 已导出绘图数据: {output_csv_signal} (共 {len(final_signal_df)} 行)")

    # 提取 IMF 统计信息
    stats_dfs = []
    for mat_id in ['001', '002', '003']:
        key = f'{mat_id}_imf_stats_table'
        if key in data:
            df_stats = pd.DataFrame(data[key])
            df_stats['Material_ID'] = mat_id
            stats_dfs.append(df_stats)

    if stats_dfs:
        final_stats_df = pd.concat(stats_dfs, ignore_index=True)
        final_stats_df.to_csv(output_csv_stats, index=False, encoding='utf-8-sig')
        print(f"  ✅ 已导出统计数据: {output_csv_stats} (共 {len(final_stats_df)} 行)")


if __name__ == "__main__":
    # 输入文件名（需放在当前目录下）
    forecast_json_file = './data/fig_forecast_curves_comparison_data.json'
    vmd_json_file = './data/fig_vmd_combined_multi_material_final_data.json'

    print("=" * 50)
    print("开始转换 JSON 数据源文件为 CSV...")
    print("=" * 50)

    # 转换预测对比数据
    if os.path.exists(forecast_json_file):
        convert_forecast_json(
            forecast_json_file,
            './data/Forecast_Comparison_Source_Data.csv'
        )
    else:
        print(f"❌ 未找到文件: {forecast_json_file}")

    # 转换 VMD 分解数据
    if os.path.exists(vmd_json_file):
        convert_vmd_json(
            vmd_json_file,
            './data/VMD_Signal_Source_Data.csv',  # 绘图主数据
            './data/VMD_IMF_Statistics_Summary.csv'  # IMF 统计信息
        )
    else:
        print(f"❌ 未找到文件: {vmd_json_file}")

    print("=" * 50)
    print("转换完成！")
