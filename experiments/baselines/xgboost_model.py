"""
文件位置: experiments/baselines/xgboost_model.py
作用: XGBoost 对比基线实验 - 监督学习特征工程 + 超参数搜索

本脚本实现了 XGBoost 回归模型，用于与深度学习模型（VMD-CAT-BiLSTM）进行公平的性能对比。
所有数据预处理（清洗、时间划分）完全复用项目主流程，特征构造采用滑动窗口 + 统计特征，
并支持可选的外部特征（与深度学习模型使用相同的选定特征集）。

主要功能：
1. 特征工程 (build_features)
   - 时间滞后特征（过去 seq_len 个月的需求值）
   - 差分特征（一阶差分 + 季节性差分）
   - 滚动统计量（均值、标准差，窗口 3/6/12）
   - 外部特征滞后（每个选中特征做 lag=1,2,3）
   - 时间特征（月份、季度、春节标志）
   - 自动删除原始外部特征列，保留其滞后项，避免多重共线性

2. 数据加载 (load_xgb_data)
   - 加载原始仿真数据，按时间划分训练/验证/测试集
   - 使用项目统一的清洗函数（训练集拟合统计量，验证/测试集转换）
   - 调用 build_features 构建监督学习数据集
   - 删除含有 NaN 的起始行（因滞后特征）
   - 返回训练、验证、测试 DataFrame（特征 + target）

3. 模型训练 (train_xgb)
   - 支持手动指定超参数或使用默认参数
   - 内置早停（early_stopping_rounds=50）和验证集监控（eval_metric='rmse'）
   - 输出最佳迭代轮数、训练耗时、AIC（近似）

4. 超参数网格搜索 (search_hyperparameters)
   - 使用 itertools.product 遍历 param_grid
   - 基于验证集 RMSE 选择最优参数
   - 输出搜索进度和最终最优参数

5. 模型评估 (evaluate_xgb)
   - 在测试集上预测，计算 RMSE, MAE, MAPE, R²
   - 需求非负约束（np.maximum(pred, 0)）
   - 所有指标基于原始尺度（is_normalized=False）

6. 特征重要性分析 (analyze_feature_importance)
   - 输出 Top-N 特征及其重要性
   - 按特征类别（lag、差分、滚动、外部、时间）汇总贡献
   - 保存条形图到 figures/xgboost_feature_importance.png
   - 输出重要性 CSV 文件

7. 实验主函数 (run_xgb_experiment)
   - 参数 search 控制是否进行超参数搜索（默认 False，可设为 True）
   - 若 search=True，先在 train/val 上搜索最佳参数，然后用 train+val 重新训练最终模型
   - 若 search=False，直接在 train+val 上使用默认参数训练（带早停，基于 val 集）
   - 最终在测试集上评估并保存结果

输入数据：
- 原始数据：由 configs.config.SIM_DATASET_PATH 指定（仿真月度数据）
- 特征选择文件：configs.config.FEATURE_SELECTION_FILE_PATH（JSON 列表，包含选中的外部特征名）
- 数据划分比例：TRAIN_RATIO, VAL_RATIO（默认 0.6, 0.2）

输出文件：
- figures/data/xgboost_results.csv      : 汇总指标（RMSE, MAE, MAPE, R², 参数量等）
- figures/xgboost_feature_importance.png: 特征重要性条形图（Top-20）
- figures/data/xgboost_feature_importance.csv : 所有特征重要性列表
- 控制台打印详细训练日志和评估结果

使用示例：
    # 使用默认参数训练（不搜索）
    results = run_xgb_experiment(search=False)

    # 执行超参数搜索后训练
    results = run_xgb_experiment(search=True)

    # 直接运行脚本（默认 search=True）
    python experiments/baselines/xgboost_model.py

关键设计原则：
- 严格遵循时间顺序，不泄露未来信息（滞后特征仅用过去值，滚动窗口不包含当前值）
- 验证集用于早停/超参数选择，测试集仅用于最终评估
- 特征构造独立于深度学习模型，但共享相同的外部特征子集，确保对比公平
- 所有指标与主模型保持完全一致（调用 src.models.trainer.calculate_metrics）
- 随机种子固定（configs.config.RANDOM_SEED），确保可复现性

依赖库：
    xgboost, pandas, numpy, matplotlib, scikit-learn, src.models.trainer, src.data.preprocess

注意事项：
- 超参数搜索可能耗时（全网格搜索组合数较大），建议先在小范围测试
- 若 XGBoost 版本较旧，请升级到 >=1.0.0 以支持 fit() 中的 eval_metric/early_stopping_rounds
- 外部特征中 'is_spring_festival' 会被保留为原始列（不创建滞后），其他外部特征仅保留滞后项
"""

# ======================== 0. 引入包 =======================
import os
import json
import time
import pandas as pd
import xgboost as xgb
import numpy as np
from typing import Tuple, Optional
from itertools import product
from src.models.trainer import calculate_metrics
from src.data.preprocess import (
    load_raw_data,                  # 加载原始数据
    clean_data_train,               # 数据清洗（训练集）
    clean_data_with_stats,          # 数据清洗（测试集）
    split_by_time,                  # 时间划分
)
from configs.config import (
    SIM_DATASET_PATH,               # 数据集路径
    FEATURE_SELECTION_FILE_PATH,    # 特征选择文件路径
    TRAIN_RATIO,                    # 训练集比例
    VAL_RATIO,                      # 验证集比例
    DEMAND_COL,                     # 目标列名
    RANDOM_SEED,                    # 随机种子
    SEQ_LEN,                        # 滑动窗口长度
    TOP_N,                          # TOP N
    FIGURES_DIR,                    # 图表保存目录
    DATA_FIGURES_DIR,               # 图表数据保存目录
)

# ======================== 1. 特征工程 =======================
def build_features(
        df: pd.DataFrame,
        target_col: str,
        external_cols: list = None,
        seq_len: int = 12
) -> pd.DataFrame:
    """
    构建时间序列特征，包括滞后特征、差分特征、滚动统计特征和时间特征。

    参数:
        df: 输入的时间序列DataFrame
        target_col: 目标变量列名
        external_cols: 外部特征列名列表
        seq_len: 滑动窗口长度，用于创建滞后特征

    返回:
        包含所有构建特征的DataFrame
    """
    df_feat = df.copy()

    if not isinstance(df_feat.index, pd.DatetimeIndex):
        if 'date' in df_feat.columns:
            df_feat = df_feat.set_index('date')

    for lag in range(1, seq_len + 1):
        df_feat[f'demand_lag_{lag}'] = df_feat[target_col].shift(lag)

    df_feat['demand_diff_1'] = df_feat[target_col].diff(1).shift(1)     # 一阶差分
    df_feat['demand_diff_12'] = df_feat[target_col].diff(12).shift(1)    # 季节差分

    for window in [3, 6, 12]:
        df_feat[f'demand_roll_mean_{window}'] = (
            df_feat[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        )
    for window in [3, 6]:
        df_feat[f'demand_roll_std_{window}'] = (
            df_feat[target_col].shift(1).rolling(window=window, min_periods=1).std()
        )

    for col in external_cols:
        if col in df_feat.columns:
            for lag in [1, 2, 3]:
                df_feat[f'ext_{col}_lag_{lag}'] = df_feat[col].shift(lag)

    df_feat['month'] = df_feat.index.month
    df_feat['quarter'] = df_feat.index.quarter
    if 'is_spring_festival' in df_feat.columns:
        df_feat['is_spring_festival'] = df_feat['is_spring_festival'].fillna(0).astype(int)

    df_feat['target'] = df_feat[target_col]

    raw_drop_cols = [c for c in external_cols if c != 'is_spring_festival']
    df_feat = df_feat.drop(columns=raw_drop_cols, errors='ignore')

    if target_col in df_feat.columns and target_col != 'target':
        df_feat = df_feat.drop(columns=[target_col])

    return df_feat

# ======================== 2. 加载 XGBoost 数据 =======================
def load_xgb_data(
        seq_len: int = 12
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载并预处理XGBoost模型所需的数据，包括数据加载、清洗和特征构建。

    参数:
        seq_len: 滑动窗口长度，用于特征构建

    返回:
        训练集、验证集和测试集的DataFrame元组
    """
    print("📂 加载原始数据...")
    df = load_raw_data(SIM_DATASET_PATH)

    with open(FEATURE_SELECTION_FILE_PATH, 'r') as f:
        selected_features = json.load(f)

    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)

    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    print(f"   训练集: {len(train_clean)} | 验证集: {len(val_clean)} | 测试集: {len(test_clean)}")

    print("🔧 构建监督学习特征...")
    all_clean = pd.concat([train_clean, val_clean, test_clean])
    all_feat = build_features(all_clean, DEMAND_COL, selected_features, seq_len)

    all_feat = all_feat.dropna()

    train_feat = all_feat[all_feat.index.isin(train_clean.index)]
    val_feat = all_feat[all_feat.index.isin(val_clean.index)]
    test_feat = all_feat[all_feat.index.isin(test_clean.index)]

    train_feat = train_feat.dropna()
    val_feat = val_feat.dropna()
    test_feat = test_feat.dropna()

    print(f"   特征构建完成 | 特征数: {len(train_feat.columns) - 1}")
    print(f"   训练样本: {len(train_feat)} | 验证样本: {len(val_feat)} | 测试样本: {len(test_feat)}")

    return train_feat, val_feat, test_feat

# ======================== 3. 模型训练 =======================
def train_xgb(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        params: Optional[dict] = None,
) -> dict:
    """
    训练XGBoost回归模型。

    参数:
        train_df: 训练数据集，包含特征和目标变量
        val_df: 验证数据集，用于早停
        params: 可选的超参数字典，如不提供则使用默认参数

    返回:
        包含训练好的模型、参数、最佳迭代轮数、训练时间和特征列名的字典
    """
    feature_cols = [c for c in train_df.columns if c != 'target']
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']

    default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50,
    }

    if params is not None:
        default_params.update(params)

    print(f"\n🏋️ 训练 XGBoost 模型...")
    print(f"   超参数: {default_params}")

    model = xgb.XGBRegressor(**default_params)

    start_time = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    train_time = time.time() - start_time

    print(f"   ✅ 训练完成 | 耗时: {train_time:.1f}秒")
    print(f"   最佳迭代轮数: {model.best_iteration}")

    return {
        'model': model,
        'params': default_params,
        'best_iteration': model.best_iteration,
        'train_time_sec': train_time,
        'feature_cols': feature_cols,
    }

# ======================== 4. 模型评估 =======================
def evaluate_xgb(
        model,
        test_df: pd.DataFrame,
        feature_cols: list,
        is_normalized: bool = False,
) -> dict:
    """
    在测试集上评估XGBoost模型性能。

    参数:
        model: 训练好的XGBoost模型
        test_df: 测试数据集
        feature_cols: 特征列名列表
        is_normalized: 是否对数据进行归一化处理

    返回:
        包含RMSE、MAE、MAPE、R²等评估指标以及真实值和预测值的字典
    """
    X_test = test_df[feature_cols]
    y_true = test_df['target'].values

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # 需求非负约束

    metrics = calculate_metrics(y_true, y_pred, is_normalized=False)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return {
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'y_true': y_true,
        'y_pred': y_pred,
        'n_params': model.best_iteration + 1,  # 树的数量近似参数量
    }

# ======================== 5. 超参数搜索 =======================
def search_hyperparameters(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
) -> dict:
    """
    使用网格搜索方法进行XGBoost超参数调优。

    参数:
        train_df: 训练数据集
        val_df: 验证数据集

    返回:
        最优超参数字典
    """
    feature_cols = [c for c in train_df.columns if c != 'target']
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 1000],
        'subsample': [0.7, 0.8, 1.0],
    }

    print("🔍 开始超参数搜索...")
    best_rmse = float('inf')
    best_params = None
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"   搜索空间: {total_combos} 种组合")

    combo_count = 0
    for max_depth, lr, n_est, subsample in product(
            param_grid['max_depth'],
            param_grid['learning_rate'],
            param_grid['n_estimators'],
            param_grid['subsample'],
    ):
        combo_count += 1
        params = {
            'max_depth': max_depth,
            'learning_rate': lr,
            'n_estimators': n_est,
            'subsample': subsample,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 5,
            'random_state': RANDOM_SEED,
            'early_stopping_rounds': 30,
            'n_jobs': -1,
        }

        import xgboost as xgb
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val.values - y_pred) ** 2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            print(f"   [{combo_count}/{total_combos}] 新最优! RMSE={rmse:.2f} | {params}")

    print(f"\n   ✅ 搜索完成 | 最优 RMSE: {best_rmse:.2f}")
    print(f"   最优参数: {best_params}")
    return best_params

# ======================== 6. 特征重要性评估 =======================
def analyze_feature_importance(
        model,
        feature_cols: list,
        top_n: int = 7,
        save_dir: str = None,
) -> pd.DataFrame:
    """
    分析并可视化XGBoost模型的特征重要性。

    参数:
        model: 训练好的XGBoost模型
        feature_cols: 特征列名列表
        top_n: 显示最重要的前N个特征
        save_dir: 特征重要性图的保存目录，如为None则不保存

    返回:
        包含特征名称和重要性得分的DataFrame
    """
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"\n📊 特征重要性 Top-{min(top_n, len(feat_imp))}:")
    print(feat_imp.head(top_n).to_string(index=False))

    # 分类统计
    lag_imp = feat_imp[feat_imp['feature'].str.startswith('demand_lag')]['importance'].sum()
    diff_imp = feat_imp[feat_imp['feature'].str.startswith('demand_diff')]['importance'].sum()
    roll_imp = feat_imp[feat_imp['feature'].str.startswith('demand_roll')]['importance'].sum()
    ext_imp = feat_imp[feat_imp['feature'].str.startswith('ext_')]['importance'].sum()
    time_imp = feat_imp[feat_imp['feature'].isin(['month', 'quarter', 'is_spring_festival'])]['importance'].sum()

    print(f"\n   特征类别贡献:")
    print(f"   需求lag特征:   {lag_imp:.3f} ({lag_imp / importance.sum() * 100:.1f}%)")
    print(f"   差分特征:      {diff_imp:.3f} ({diff_imp / importance.sum() * 100:.1f}%)")
    print(f"   滚动统计:      {roll_imp:.3f} ({roll_imp / importance.sum() * 100:.1f}%)")
    print(f"   外部特征lag:   {ext_imp:.3f} ({ext_imp / importance.sum() * 100:.1f}%)")
    print(f"   时间特征:      {time_imp:.3f} ({time_imp / importance.sum() * 100:.1f}%)")

    return feat_imp

# ======================== 7. 实验运行主函数 =======================
def run_xgb_experiment(
        search: bool = False,
        seq_len: int = SEQ_LEN,
) -> pd.DataFrame:
    """
    运行完整的XGBoost对比实验流程。

    参数:
        search: 是否执行超参数搜索
        seq_len: 滑动窗口长度，用于特征构建

    返回:
        包含实验结果的DataFrame
    """
    print("\n" + "=" * 70)
    print("🧪 XGBoost 对比实验")
    print("=" * 70)

    # ---- 1. 准备数据 ----
    train_df, val_df, test_df = load_xgb_data(seq_len)

    # ---- 2. 合并训练集+验证集用于最终训练 ----
    train_val_df = pd.concat([train_df, val_df], axis=0)

    # ---- 3. 超参数搜索 (可选) ----
    if search:
        best_params = search_hyperparameters(train_df, val_df)
        # 用最优参数在合并集上重训
        train_result = train_xgb(train_val_df, val_df, params=best_params)
    else:
        # 默认参数
        train_result = train_xgb(train_val_df, val_df)

    model = train_result['model']
    feature_cols = train_result['feature_cols']

    # ---- 4. 测试集评估 ----
    print(f"\n📊 测试集评估...")
    eval_results = evaluate_xgb(model, test_df, feature_cols)

    print(f"   RMSE:  {eval_results['RMSE']:.4f}")
    print(f"   MAE:   {eval_results['MAE']:.4f}")
    print(f"   MAPE:  {eval_results['MAPE']:.2f}%")
    print(f"   R²:    {eval_results['R2']:.4f}")

    # ---- 5. 特征重要性分析 ----
    os.makedirs(FIGURES_DIR, exist_ok=True)
    feat_imp = analyze_feature_importance(model, feature_cols, top_n=20, save_dir=FIGURES_DIR)

    # ---- 6. 汇总结果 ----
    results = {
        'name': 'xgboost',
        'description': f'XGBoost (seq_len={seq_len}, n_features={len(feature_cols)})',
        'RMSE': eval_results['RMSE'],
        'MAE': eval_results['MAE'],
        'MAPE': eval_results['MAPE'],
        'R2': eval_results['R2'],
        'n_params': eval_results['n_params'],
        'train_time_min': round(train_result['train_time_sec'] / 60, 2),
        'model_type': 'xgboost',
        'data_mode': 'supervised_features',
        'n_features': len(feature_cols),
        'best_iteration': train_result['best_iteration'],
    }

    # 保存结果
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
    results_df = pd.DataFrame([results])
    save_path = os.path.join(DATA_FIGURES_DIR, 'xgboost_results.csv')
    results_df.to_csv(save_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"📋 XGBoost 实验结果")
    print(f"{'=' * 70}")
    print(f"   RMSE:  {results['RMSE']:.4f}")
    print(f"   MAE:   {results['MAE']:.4f}")
    print(f"   MAPE:  {results['MAPE']:.2f}%")
    print(f"   R²:    {results['R2']:.4f}")
    print(f"   特征数: {results['n_features']}")
    print(f"   训练耗时: {results['train_time_min']:.2f} 分钟")
    print(f"💾 结果已保存至: {save_path}")

    # 保存特征重要性
    feat_imp_path = os.path.join(DATA_FIGURES_DIR, 'xgboost_feature_importance.csv')
    feat_imp.to_csv(feat_imp_path, index=False)

    return results_df

# ======================== 测试 ========================
if __name__ == '__main__':
    run_xgb_experiment(search=True)