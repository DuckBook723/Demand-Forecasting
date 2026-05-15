"""
文件位置: src/features/feature_selector.py
作用: 双重特征选择——斯皮尔曼相关性初筛 + 随机森林重要性精筛
所包含的函数:
  - load_train_data(): 加载训练集数据，划分x和y
  - spearman_correlation_analysis(): 斯皮尔曼相关性分析
  - random_forest_importance(): 随机森林特征重要性评估
  - dual_feature_select(): 执行双重筛选并保存结果
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 读取 data/processed/train.csv
  - 输出 data/features/selected_features.json, data/features/feature_evaluation.csv
"""

# ======================== 0. 引入包 =======================
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

# ======================== 1. 加载训练集数据，划分x和y =======================
def load_train_data(path: str, demand_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载训练数据并分离特征和目标变量。

    从指定的CSV文件路径读取数据，自动解析'date'列为日期类型并将其设置为索引，
    然后将数据分为特征集(X)和目标变量(y)两部分。

    Args:
        path (str): CSV文件的路径，文件需包含'date'列和需求列。
        demand_col (str): 需求目标列的列名，将从数据中分离出来作为目标变量。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 包含两个元素的元组：
            - df_x: 删除了需求列后的特征DataFrame
            - df_y: 仅包含需求列的目标变量Series
    """
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

# ======================== 2. 斯皮尔曼相关性分析 =======================
def spearman_correlation_analysis(x: pd.DataFrame, y: pd.Series, threshold: float) -> tuple[pd.Series, np.ndarray]:
    """
    基于Spearman相关系数进行特征相关性分析。

    计算每个特征变量与目标变量之间的Spearman秩相关系数，
    并根据给定的阈值筛选出高相关性的特征（支持处理缺失值）。

    Args:
        x (pd.DataFrame): 特征数据框，包含待分析的特征变量。
        y (pd.Series): 目标变量序列。
        threshold (float): 相关性阈值，用于筛选高相关性特征（基于绝对值）。

    Returns:
        tuple[pd.Series, np.ndarray]: 包含两个元素的元组：
            - corr_series: 每个特征与目标变量的Spearman相关系数Series，索引为特征名
            - mask: 布尔数组，标识哪些特征的绝对相关系数超过阈值
    """
    cols = x.columns
    corrs = [spearmanr(x[col], y, nan_policy='omit')[0] for col in cols]
    corr_series = pd.Series(corrs, index=cols)
    mask = np.abs(corrs) > threshold
    return corr_series, mask

# ======================== 3. 随机森林特征重要性评估 =======================
def random_forest_importance(x: pd.DataFrame, y: pd.Series, mask: np.ndarray[bool], top_n: int) -> tuple[pd.Series, list]:
    """
    基于随机森林模型计算特征重要性并筛选Top N重要特征。

    使用随机森林回归模型对已通过初步筛选的特征进行训练，
    根据特征重要性评分选择最重要的前N个特征。

    Args:
        x (pd.DataFrame): 特征数据框，包含所有待评估的特征变量。
        y (pd.DataFrame): 目标变量数据框。
        mask (np.ndarray[bool]): 布尔掩码数组，标识哪些特征已通过初步筛选。
        top_n (int): 需要选择的top N个最重要特征的数量。

    Returns:
        tuple[np.ndarray[bool], pd.Series]: 包含两个元素的元组：
            - mask: 新的布尔掩码数组，仅标记Top N重要特征为True
            - importances: 每个特征的重要性评分Series，索引为特征名
    """
    selected_cols = x.columns[mask]
    # print(f"🚀 开始特征筛选 | 原始特征数: {len(x.columns)}")
    # print(f"✅ 特征筛选完成 | 选取特征数: {len(selected_cols)}")
    # print(x[selected_cols])
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(x[selected_cols], y)

    importances = pd.Series(rf.feature_importances_, index=selected_cols)

    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

    return (
        importances,
        top_features
    )

# ======================== 4. 双重特征筛选并保存结果 =======================
def dual_feature_select(x: pd.DataFrame, y: pd.Series, threshold: float, top_n: int) -> None:
    """
    执行双重特征选择流程，结合Spearman相关性分析和随机森林重要性评估。

    首先使用Spearman相关系数进行初步筛选，去除与目标变量相关性较低的特征；
    然后在剩余特征上应用随机森林模型，根据特征重要性评分选择Top-N个最重要特征。
    最终将筛选结果和评估指标保存到指定目录。

    Args:
        x (pd.DataFrame): 特征数据框，包含所有待筛选的特征变量。
        y (pd.Series): 目标变量序列。
        threshold (float): Spearman相关性阈值（绝对值），用于初步筛选特征。
        top_n (int): 最终需要选择的Top N个最重要特征的数量。

    Returns:
        None: 该函数不返回值，但会将结果保存到文件系统中。
    """
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

# ========================= 测试 =======================
if __name__ == '__main__':
    # 1. 加载数据
    x, y = load_train_data(TRAIN_FILE_PATH, DEMAND_COL)
    print(f"✅ 训练集数据已加载 | x: {x.shape} | y: {y.shape}")

    # 2. 执行双重筛选并保存
    final_features = dual_feature_select(x, y, SPEARMAN_THRESHOLD, TOP_N)
    print("\n🎉 特征工程阶段完成！")
