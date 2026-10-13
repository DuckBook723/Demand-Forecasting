"""
双重特征筛选：斯皮尔曼相关系数初筛 + 随机森林重要性精筛，取 Top-N 特征。

读取 data/processed/train.csv，输出 selected_features.json 与 feature_evaluation.csv。
"""

# ---- 引入依赖 ----
import os
import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from configs.config import (
    DEMAND_COL,                         # 需求列
    TRAIN_FILE_PATH,                    # 训练集文件路径
    SPEARMAN_THRESHOLD,                 # 斯皮尔曼相关性初筛阈值
    FEATURES_DIR,                       # 特征保存目录
    FIGURES_DIR,                        # 绘图保存目录
    TOP_N,                              # 随机森林特征重要性评估的top N
    FEATURE_SELECTION_FILE_PATH,        # 特征保存文件路径
    FEATURE_EVALUATION_FILE_PATH,       # 特征评估保存文件路径
)

# ---- 加载训练集，划分 X 和 y ----
def load_train_data(path: str, demand_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 CSV（date 作索引），拆出特征 X 和目标 y；IMF 与残差列一并剔除。"""
    df = pd.read_csv(path, parse_dates=['date'], index_col='date')

    cols2drop = [demand_col]

    for col in df.columns:
        if col.startswith('IMF') or col.lower().startswith('residual'):
            cols2drop.append(col)

    cols2drop = [col for col in cols2drop if col in df.columns]

    df_x = df.drop(columns=cols2drop)
    df_y = df[demand_col]
    return (
        df_x,
        df_y
    )

# ---- 斯皮尔曼相关性初筛 ----
def spearman_correlation_analysis(x: pd.DataFrame, y: pd.Series, threshold: float) -> tuple[pd.Series, np.ndarray]:
    """计算各特征与目标的 Spearman 秩相关系数（缺失值跳过），返回系数与 |ρ| 超阈值的掩码。"""
    cols = x.columns
    corrs = [spearmanr(x[col], y, nan_policy='omit')[0] for col in cols]
    corr_series = pd.Series(corrs, index=cols)
    mask = np.abs(corrs) > threshold
    return corr_series, mask

# ---- 随机森林重要性精筛 ----
def random_forest_importance(x: pd.DataFrame, y: pd.Series, mask: np.ndarray[bool], top_n: int) -> tuple[pd.Series, list]:
    """在初筛通过的特征上训练随机森林，返回重要性评分与按评分选出的 Top-N 特征名。"""
    selected_cols = x.columns[mask]
    # print(f"开始特征筛选 | 原始特征数: {len(x.columns)}")
    # print(f"特征筛选完成 | 选取特征数: {len(selected_cols)}")
    # print(x[selected_cols])
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(x[selected_cols], y)

    importances = pd.Series(rf.feature_importances_, index=selected_cols)

    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

    return (
        importances,
        top_features
    )

# ---- 双重筛选并保存结果 ----
def dual_feature_select(x: pd.DataFrame, y: pd.Series, threshold: float, top_n: int) -> None:
    """串联两个筛选步骤，把最终特征列表和评估表（Spearman 系数 + RF 重要性）写入磁盘。"""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    corr_series, mask = spearman_correlation_analysis(x, y, threshold)
    print(f"🔍 斯皮尔曼初筛：原始特征数 {len(x.columns)}，通过阈值|ρ|>{threshold} 的特征数 {sum(mask)}")

    importances, selected_features = random_forest_importance(x, y, mask, top_n)
    print(f"🌲 随机森林精筛：最终锁定 Top-{top_n} 特征: {selected_features}")

    with open(FEATURE_SELECTION_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(selected_features, f, ensure_ascii=False, indent=4)
    print(f"💾 特征列表已保存至: {FEATURE_SELECTION_FILE_PATH}")

    eval_df = pd.DataFrame({
        'Spearman_Corr': corr_series,
        'RF_Importance': importances
    }).fillna(0).sort_values(by='RF_Importance', ascending=False)

    eval_df.to_csv(FEATURE_EVALUATION_FILE_PATH)
    print(f"💾 特征评估已保存至: {FEATURE_EVALUATION_FILE_PATH}")

# ---- 测试 ----
if __name__ == '__main__':
    # 1. 加载数据
    x, y = load_train_data(TRAIN_FILE_PATH, DEMAND_COL)
    print(f"✅ 训练集数据已加载 | x: {x.shape} | y: {y.shape}")

    # 2. 执行双重筛选并保存
    final_features = dual_feature_select(x, y, SPEARMAN_THRESHOLD, TOP_N)
    print("\n🎉 特征工程阶段完成！")