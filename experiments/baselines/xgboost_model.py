"""
文件位置: experiments/baselines/xgboost_model_v3.py
作用: XGBoost 对比基线实验 — XGBoost 3.x 完全兼容版（底层 train API）

修复清单:
1. [P0] 兼容 XGBoost 3.2.0：使用 xgboost.train() 替代 XGBRegressor.fit()
2. [P0] 全局随机种子固定（Python/NumPy/XGBoost）
3. [P0] 修复验证集泄露：阶段一找 best_iteration，阶段二固定轮数重训
4. [P1] 单线程确定性训练（nthread=1）
"""

import os
import sys
import random
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple, Optional
from itertools import product
from sklearn.preprocessing import StandardScaler

from src.models.trainer import calculate_metrics
from src.data.preprocess import (
    load_raw_data, clean_data_train, clean_data_with_stats, split_by_time
)
from configs.config import (
    SIM_DATASET_PATH, FEATURE_SELECTION_FILE_PATH,
    TRAIN_RATIO, VAL_RATIO, DEMAND_COL, RANDOM_SEED,
    SEQ_LEN, FIGURES_DIR, DATA_FIGURES_DIR
)

warnings.filterwarnings("ignore")

# ==================== 全局随机种子固定 ====================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    print(f"🔒 全局随机种子已固定: seed={seed}")
    print(f"   XGBoost 版本: {xgb.__version__}")
    print(f"   NumPy 版本: {np.__version__}")


# ==================== 底层 XGBoost 训练核心 ====================
def _train_xgb_core(
    X_train, y_train, X_val, y_val,
    params: dict, num_boost_round: int,
    early_stopping_rounds: Optional[int] = None
):
    """
    使用 xgboost.train() 底层 API 训练，兼容 XGBoost 1.x/2.x/3.x。

    返回:
        model: 包装后的 XGBRegressor（支持 predict/feature_importances_）
        best_iteration: 最佳迭代轮数
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 构建 xgboost.train() 参数
    train_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': params.get('random_state', 42),
        'tree_method': params.get('tree_method', 'hist'),
        'nthread': 1,  # 单线程确定性
    }

    # 复制其他超参数
    for key in ['max_depth', 'learning_rate', 'subsample', 'colsample_bytree',
                'reg_alpha', 'reg_lambda', 'min_child_weight']:
        if key in params:
            train_params[key] = params[key]

    evals = [(dtrain, 'train'), (dval, 'validation')]

    if early_stopping_rounds is not None:
        callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
        bst = xgb.train(
            train_params, dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=False,
        )
        best_iteration = bst.num_boosted_rounds()
    else:
        bst = xgb.train(
            train_params, dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            verbose_eval=False,
        )
        best_iteration = num_boost_round

    # 包装为 XGBRegressor（保持 sklearn 接口兼容性）
    sklearn_params = {k: v for k, v in params.items()
                     if k not in ['eval_metric', 'early_stopping_rounds', 'n_jobs']}
    sklearn_params['n_estimators'] = best_iteration

    model = xgb.XGBRegressor(**sklearn_params)
    model._Booster = bst
    model.n_estimators = best_iteration

    return model, best_iteration


# ==================== 1. 特征工程 ====================
def build_features(df: pd.DataFrame, target_col: str, external_cols: list = None, seq_len: int = 12) -> pd.DataFrame:
    df_feat = df.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        if 'date' in df_feat.columns:
            df_feat = df_feat.set_index('date')

    for lag in range(1, seq_len + 1):
        df_feat[f'demand_lag_{lag}'] = df_feat[target_col].shift(lag)

    df_feat['demand_diff_1'] = df_feat[target_col].diff(1).shift(1)
    df_feat['demand_diff_12'] = df_feat[target_col].diff(12).shift(1)

    for window in [3, 6, 12]:
        df_feat[f'demand_roll_mean_{window}'] = (
            df_feat[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        )
    for window in [3, 6]:
        df_feat[f'demand_roll_std_{window}'] = (
            df_feat[target_col].shift(1).rolling(window=window, min_periods=1).std()
        )

    for col in external_cols or []:
        if col in df_feat.columns:
            for lag in [1, 2, 3]:
                df_feat[f'ext_{col}_lag_{lag}'] = df_feat[col].shift(lag)

    df_feat['month'] = df_feat.index.month
    df_feat['quarter'] = df_feat.index.quarter
    if 'is_spring_festival' in df_feat.columns:
        df_feat['is_spring_festival'] = df_feat['is_spring_festival'].fillna(0).astype(int)

    df_feat['target'] = df_feat[target_col]

    if target_col in df_feat.columns:
        df_feat = df_feat.drop(columns=[target_col])

    raw_drop_cols = [c for c in (external_cols or []) if c != 'is_spring_festival']
    df_feat = df_feat.drop(columns=raw_drop_cols, errors='ignore')

    return df_feat


# ==================== 2. 数据加载 ====================
def load_xgb_data(seq_len: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("📂 加载原始数据...")
    df = load_raw_data(SIM_DATASET_PATH)

    with open(FEATURE_SELECTION_FILE_PATH, 'r') as f:
        selected_features = json.load(f)

    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    print(f"   训练集: {len(train_clean)} | 验证集: {len(val_clean)} | 测试集: {len(test_clean)}")

    all_clean = pd.concat([train_clean, val_clean, test_clean])
    all_feat = build_features(all_clean, DEMAND_COL, selected_features, seq_len)
    all_feat = all_feat.dropna()

    train_feat = all_feat[all_feat.index.isin(train_clean.index)]
    val_feat = all_feat[all_feat.index.isin(val_clean.index)]
    test_feat = all_feat[all_feat.index.isin(test_clean.index)]

    print(f"   特征数: {len(train_feat.columns) - 1}")
    print(f"   训练: {len(train_feat)} | 验证: {len(val_feat)} | 测试: {len(test_feat)}")

    return train_feat, val_feat, test_feat


# ==================== 3. 模型训练（修复泄露 + 兼容 3.x）====================
def train_xgb(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, params: Optional[dict] = None) -> dict:
    feature_cols = [c for c in train_df.columns if c != 'target']
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values

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
        'tree_method': 'hist',
    }
    if params is not None:
        default_params.update(params)

    print(f"\n🏋️ 训练 XGBoost...")
    print(f"   超参数: { {k:v for k,v in default_params.items()} }")

    best_iteration = default_params['n_estimators']

    # 阶段一：用验证集早停找 best_iteration
    if val_df is not None and len(val_df) > 0:
        X_val = val_df[feature_cols].values
        y_val = val_df['target'].values

        stage1_model, best_iteration = _train_xgb_core(
            X_train, y_train, X_val, y_val,
            default_params, default_params['n_estimators'],
            early_stopping_rounds=50
        )
        print(f"   ✅ 早停完成 | 最佳迭代: {best_iteration}")

    # 阶段二：最终训练（固定轮数，不传入验证集，防止泄露）
    print(f"   🔒 最终训练: 固定 n_estimators={best_iteration}")
    final_model, _ = _train_xgb_core(
        X_train, y_train, X_train, y_train,  # 虚拟验证集（实际不用）
        default_params, best_iteration,
        early_stopping_rounds=None
    )

    return {
        'model': final_model,
        'params': default_params,
        'best_iteration': best_iteration,
        'feature_cols': feature_cols,
    }


# ==================== 4. 模型评估 ====================
def evaluate_xgb(model, test_df: pd.DataFrame, feature_cols: list) -> dict:
    X_test = test_df[feature_cols].values
    y_true = test_df['target'].values
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    metrics = calculate_metrics(y_true, y_pred, is_normalized=False)

    return {
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'y_true': y_true,
        'y_pred': y_pred,
        'n_params': model.n_estimators * len(feature_cols),  # 近似
    }


# ==================== 5. 超参数搜索（确定化 + 兼容 3.x）====================
def search_hyperparameters(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    feature_cols = [c for c in train_df.columns if c != 'target']
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 1000],
        'subsample': [0.7, 0.8, 1.0],
    }

    print("🔍 开始超参数搜索...")
    best_rmse = float('inf')
    best_params = None
    total_combos = np.prod([len(v) for v in param_grid.values()])
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
            'tree_method': 'hist',
        }

        # 【修复】使用底层 API 进行早停训练
        model, best_iter = _train_xgb_core(
            X_train, y_train, X_val, y_val,
            params, n_est, early_stopping_rounds=30
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params.copy()
            best_params['best_iteration'] = best_iter
            print(f"   [{combo_count}/{total_combos}] 新最优! RMSE={rmse:.2f} | depth={max_depth}, lr={lr}, iter={best_iter}")

    print(f"\n   ✅ 搜索完成 | 最优 RMSE: {best_rmse:.2f}")
    return best_params


# ==================== 6. 特征重要性 ====================
def analyze_feature_importance(model, feature_cols: list, top_n: int = 20, save_dir: str = None) -> pd.DataFrame:
    # 【修复】从 Booster 获取特征重要性
    if hasattr(model, '_Booster') and model._Booster is not None:
        importance_dict = model._Booster.get_score(importance_type='weight')
        # 补齐缺失特征（重要性为 0）
        importance = np.array([importance_dict.get(f'f{i}', 0.0) for i in range(len(feature_cols))])
    else:
        importance = model.feature_importances_

    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"\n📊 特征重要性 Top-{min(top_n, len(feat_imp))}:")
    print(feat_imp.head(top_n).to_string(index=False))

    categories = {
        '需求lag': feat_imp[feat_imp['feature'].str.startswith('demand_lag')]['importance'].sum(),
        '差分': feat_imp[feat_imp['feature'].str.startswith('demand_diff')]['importance'].sum(),
        '滚动统计': feat_imp[feat_imp['feature'].str.startswith('demand_roll')]['importance'].sum(),
        '外部特征lag': feat_imp[feat_imp['feature'].str.startswith('ext_')]['importance'].sum(),
        '时间特征': feat_imp[feat_imp['feature'].isin(['month', 'quarter', 'is_spring_festival'])]['importance'].sum(),
    }

    total_imp = importance.sum()
    print(f"\n   特征类别贡献:")
    for cat_name, cat_imp in categories.items():
        if cat_imp > 0:
            print(f"   {cat_name:12s}: {cat_imp:.3f} ({cat_imp / total_imp * 100:.1f}%)")

    if save_dir is not None:
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        plot_df = feat_imp.head(top_n).sort_values('importance', ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(plot_df['feature'], plot_df['importance'], color='#3498db', edgecolor='white')
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'XGBoost Top-{top_n} Feature Importance', fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'xgboost_feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 特征重要性图已保存: {save_path}")

        csv_path = os.path.join(DATA_FIGURES_DIR, 'xgboost_feature_importance.csv')
        os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
        feat_imp.to_csv(csv_path, index=False)

    return feat_imp


# ==================== 7. 实验主函数 ====================
def run_xgb_experiment(search: bool = False, seq_len: int = SEQ_LEN) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("🧪 XGBoost 对比实验（XGBoost 3.x 兼容版）")
    print("=" * 70)

    train_df, val_df, test_df = load_xgb_data(seq_len)
    train_val_df = pd.concat([train_df, val_df], axis=0)

    if search:
        best_params = search_hyperparameters(train_df, val_df)
        best_iteration = best_params.pop('best_iteration', 500)
    else:
        temp_result = train_xgb(train_df, val_df)
        best_iteration = temp_result['best_iteration']
        best_params = {}

    print(f"\n🔒 最终训练: 合并集 ({len(train_val_df)} 样本)，固定 n_estimators={best_iteration}")
    best_params['n_estimators'] = best_iteration
    train_result = train_xgb(train_val_df, val_df=None, params=best_params)

    model = train_result['model']
    feature_cols = train_result['feature_cols']

    print(f"\n📊 测试集评估...")
    eval_results = evaluate_xgb(model, test_df, feature_cols)

    print(f"   RMSE:  {eval_results['RMSE']:.4f}")
    print(f"   MAE:   {eval_results['MAE']:.4f}")
    print(f"   MAPE:  {eval_results['MAPE']:.2f}%")
    print(f"   R²:    {eval_results['R2']:.4f}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    feat_imp = analyze_feature_importance(model, feature_cols, top_n=20, save_dir=FIGURES_DIR)

    results = {
        'name': 'xgboost_v3',
        'description': f'XGBoost_v3 (seq_len={seq_len}, n_features={len(feature_cols)})',
        'RMSE': eval_results['RMSE'],
        'MAE': eval_results['MAE'],
        'MAPE': eval_results['MAPE'],
        'R2': eval_results['R2'],
        'n_params': eval_results['n_params'],
        'train_time_min': 0.0,
        'model_type': 'xgboost',
        'data_mode': 'supervised_features',
        'n_features': len(feature_cols),
        'best_iteration': train_result['best_iteration'],
    }

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
    print(f"💾 结果已保存至: {save_path}")

    return results_df


# ==================== 主入口 ====================
if __name__ == '__main__':
    seed_everything(seed=42)
    results = run_xgb_experiment(search=True)

    if not results.empty:
        log_path = os.path.join(DATA_FIGURES_DIR, 'xgboost_experiment_log.json')
        log_data = {
            "seed": 42,
            "xgboost_version": xgb.__version__,
            "numpy_version": np.__version__,
            "search": True,
            "best_r2": float(results.iloc[0]['R2']),
            "best_rmse": float(results.iloc[0]['RMSE']),
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"\n📝 实验日志已保存: {log_path}")