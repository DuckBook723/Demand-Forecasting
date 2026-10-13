# -*- coding: utf-8 -*-
"""提取绘图所需的源数据，分别生成 VMD 分解与预测对比两张图对应的 CSV 文件。"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(os.getcwd())
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import configs.config as cfg
    from src.data.preprocess import load_raw_data, clean_data_train, clean_data_with_stats, split_by_time
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

OUTPUT_DIR = PROJECT_ROOT / "exported_plot_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- 核心提取函数 ----

def export_merged_vmd_data():
    """提取三种物资的 VMD 分解数据并合并为一个 CSV，
    列为 Material_ID、Time_Step、Original_Demand 与四个 IMF 分量。"""
    print("\n[Task 1] 正在提取并合并 VMD 数据...")

    try:
        from vmdpy import VMD
    except ImportError:
        print("⚠️ 缺少 vmdpy，跳过。")
        return

    MATERIAL_IDS = ["001", "002", "003"]
    VMD_K = 4
    VMD_ALPHA = 2000
    all_data_list = []

    for mat_id in MATERIAL_IDS:
        try:
            # 切换物资配置并加载数据
            os.environ["MATERIAL_SCENARIO"] = mat_id
            import importlib;
            importlib.reload(cfg)

            df_raw = load_raw_data(cfg.SIM_DATASET_PATH)
            train_raw, val_raw, test_raw = split_by_time(df_raw, cfg.TRAIN_RATIO, cfg.VAL_RATIO)
            full_df = pd.concat([train_raw, val_raw, test_raw])
            demand_signal = full_df[cfg.DEMAND_COL].values.astype(float)

            # VMD 计算
            u, _, _ = VMD(demand_signal, VMD_ALPHA, 0, VMD_K, 0, 1, 1e-7)

            # 对齐长度
            imfs = u
            if imfs.shape[1] < len(demand_signal):
                pad_len = len(demand_signal) - imfs.shape[1]
                imfs = np.pad(imfs, ((0, 0), (0, pad_len)), mode='edge')

            # 构建当前物资的 DataFrame
            mat_df = pd.DataFrame({
                'Material_ID': mat_id,
                'Time_Step': np.arange(len(demand_signal)),
                'Original_Demand': demand_signal,
                'IMF1_Trend': imfs[0, :],
                'IMF2_LowFreq': imfs[1, :],
                'IMF3_MidFreq': imfs[2, :],
                'IMF4_HighFreq': imfs[3, :]
            })
            all_data_list.append(mat_df)
            print(f"  ✅ [{mat_id}] VMD 数据提取完成，长度: {len(mat_df)}")

        except Exception as e:
            print(f"  ❌ [{mat_id}] 失败: {e}")

    # 合并并保存
    if all_data_list:
        final_df = pd.concat(all_data_list, ignore_index=True)
        save_path = OUTPUT_DIR / "VMD_Plot_Source_Data.csv"
        final_df.to_csv(save_path, index=False)
        print(f"\n🎉 VMD 数据已合并导出: {save_path} (总行数: {len(final_df)})")


def export_merged_prediction_data():
    """提取三种物资的模型预测对比数据并合并为一个 CSV，采用宽格式：
    Material_ID、Time_Step、True_Value 以及各模型的预测列。"""
    print("\n[Task 2] 正在提取并合并预测对比数据...")

    MATERIAL_IDS = ["001", "002", "003"]
    MODEL_MAP = {
        "Ours": "vmd_catbilstm_predictions.csv",
        "XGBoost": "xgboost_predictions.csv",
        "ARIMA": "arima_predictions.csv"
        # 可根据需要补充其他模型
    }

    all_data_list = []

    for mat_id in MATERIAL_IDS:
        mat_dir = Path(cfg.FIGURES_DIR) / 'data' / mat_id.lower()
        base_df = None

        # 尝试读取基准真实值 (优先从本文框架结果中提取)
        ours_path = mat_dir / MODEL_MAP["Ours"]
        if ours_path.exists():
            try:
                df_temp = pd.read_csv(ours_path)
                # 自动匹配列名
                true_col = [c for c in df_temp.columns if 'true' in c.lower()][0]
                pred_col = [c for c in df_temp.columns if 'pred' in c.lower()][0]

                base_df = pd.DataFrame({
                    'Material_ID': mat_id,
                    'Time_Step': df_temp.index,
                    'True_Value': df_temp[true_col],
                    'Ours_Pred': df_temp[pred_col]
                })
            except Exception as e:
                print(f"  ⚠️ [{mat_id}] 基准文件读取失败: {e}")
                continue
        else:
            print(f"  ⚠️ [{mat_id}] 未找到预测文件，跳过。")
            continue

        # 依次读取并合并其他模型
        for model_name, csv_name in MODEL_MAP.items():
            if model_name == "Ours": continue  # 已处理

            csv_path = mat_dir / csv_name
            if csv_path.exists():
                try:
                    df_model = pd.read_csv(csv_path)
                    pred_col = [c for c in df_model.columns if 'pred' in c.lower()][0]
                    # 对齐长度（取最小长度防止溢出）
                    min_len = min(len(base_df), len(df_model))
                    base_df[f'{model_name}_Pred'] = df_model[pred_col].values[:min_len]
                except Exception:
                    pass  # 忽略单个模型读取错误

        all_data_list.append(base_df)
        print(f"  ✅ [{mat_id}] 预测数据提取完成，列数: {len(base_df.columns)}")

    # 合并并保存
    if all_data_list:
        final_df = pd.concat(all_data_list, ignore_index=True)
        save_path = OUTPUT_DIR / "Prediction_Plot_Source_Data.csv"
        final_df.to_csv(save_path, index=False)
        print(f"\n🎉 预测数据已合并导出: {save_path} (总行数: {len(final_df)})")


# ---- 主程序 ----
if __name__ == '__main__':
    print("=" * 60)
    print("开始执行数据合并导出...")
    print("=" * 60)
    export_merged_vmd_data()
    export_merged_prediction_data()
    print("\n✅ 任务完成！请查看 exported_plot_data 目录下的两个 CSV 文件。")
