# diagnose_data_quality.py - 修复版
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from configs.config import SIM_DATASET_PATH, FEATURE_SELECTION_FILE_PATH
import json
import os


def evaluate_simulated_data():
    # 确保路径存在
    if not os.path.exists(SIM_DATASET_PATH):
        print(f"❌ 数据文件不存在: {SIM_DATASET_PATH}")
        return
    if not os.path.exists(FEATURE_SELECTION_FILE_PATH):
        print(f"❌ 特征文件不存在: {FEATURE_SELECTION_FILE_PATH}")
        return

    df = pd.read_csv(SIM_DATASET_PATH, parse_dates=['date'], index_col='date')

    print("🔍 数据质量诊断报告")
    print("=" * 60)

    # 1. 零需求比例（业务合理性）
    zero_ratio = (df['demand'] == 0).mean()
    print(f"1️⃣ 零需求比例: {zero_ratio:.1%} (合理范围: 月度<10%, 日度60-80%)")

    # 2. 需求分布（模型友好性）
    print(f"2️⃣ 需求统计: 均值={df['demand'].mean():.1f}, 标准差={df['demand'].std():.1f}")
    print(f"   范围: [{df['demand'].min():.0f}, {df['demand'].max():.0f}]")

    # 3. 特征-需求相关性（因果强度诊断）✅ 修复核心问题
    with open(FEATURE_SELECTION_FILE_PATH, 'r') as f:
        selected_features = json.load(f)

    corr_report = {}
    for feat in selected_features:
        if feat in df.columns:
            # ✅ 关键修复：先合并两列，再同步删除缺失值
            valid_data = df[[feat, 'demand']].dropna()
            if len(valid_data) < 10:  # 样本太少跳过
                print(f"   ⚠️ {feat}: 有效样本仅 {len(valid_data)} 行，跳过")
                continue
            corr, _ = spearmanr(valid_data[feat], valid_data['demand'])
            corr_report[feat] = abs(corr)

    if corr_report:
        avg_corr = np.mean(list(corr_report.values()))
        print(f"3️⃣ 特征-需求平均斯皮尔曼相关性: {avg_corr:.3f}")
        for feat, corr in corr_report.items():
            status = "⚠️ 强" if corr > 0.6 else ("✅ 中" if corr > 0.3 else "ℹ️ 弱")
            print(f"   {feat}: {corr:.3f} {status}")
    else:
        print("3️⃣ 特征-需求相关性: 无法计算（无有效特征）")
        avg_corr = 0

    # 4. 时序自相关性（适合时序建模）
    df_clean = df['demand'].dropna()
    if len(df_clean) > 12:
        acf_1 = df_clean.autocorr(lag=1)
        acf_12 = df_clean.autocorr(lag=12)
        print(f"4️⃣ 需求自相关性: lag-1={acf_1:.3f}, lag-12={acf_12:.3f} (适合时序建模: >0.3)")
    else:
        print("4️⃣ 需求自相关性: 样本不足，无法计算")

    # 5. 缺失值/异常值比例（噪声模拟）
    missing_ratio = df['demand'].isna().mean()
    print(f"5️⃣ 缺失值比例: {missing_ratio:.1%} (目标: ~3%)")

    print("=" * 60)

    # 综合建议
    if avg_corr > 0.6:
        print("💡 建议：因果强度可能过强，考虑降低CAUSAL_SENSITIVITY系数")
    elif avg_corr < 0.2:
        print("💡 建议：特征贡献可能被噪声淹没，考虑增强因果调制")
    else:
        print("✅ 数据质量良好，可直接用于实验")


if __name__ == '__main__':
    evaluate_simulated_data()