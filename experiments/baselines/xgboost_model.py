"""XGBoost 对比基线，直接使用全部 11 维原始外部特征，不依赖特征选择文件，
由模型自行评估各特征重要性，兼容单物资单区域数据格式。"""

import os
import random
import json
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import product
from sklearn.preprocessing import StandardScaler

from src.models.trainer import calculate_metrics
from src.data.preprocess import load_raw_data, clean_data_train, clean_data_with_stats, split_by_time
from configs.config import (
    SIM_DATASET_PATH, TRAIN_RATIO, VAL_RATIO,
    DEMAND_COL, RANDOM_SEED, SEQ_LEN, FIGURES_DIR, DATA_FIGURES_DIR,
)

warnings.filterwarnings("ignore")


# ---- 原始外部特征 ----
# 与 generate_data_enhanced.py 输出列完全一致(单物资单区域版)
RAW_EXTERNAL_COLS = [
    "temperature",
    "extreme_weather_days",
    "wind_speed",
    "macro_invest_growth",
    "gdp_growth",
    "copper_price_index",
    "power_load_MW",
    "equipment_aging_rate",
    "bidding_intensity",
    "renewable_capacity_MW",
    "is_spring_festival",
]


# ---- 全局随机种子 ----
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    print(f"🔒 全局随机种子已固定: seed={seed}")
    print(f"   XGBoost 版本: {xgb.__version__}")
    print(f"   NumPy 版本: {np.__version__}")


# ---- 底层训练 ----
def _train_xgb_core(X_train, y_train, X_val, y_val, params, num_boost_round, early_stopping_rounds=None):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    train_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": params.get("random_state", 42),
        "tree_method": params.get("tree_method", "hist"),
        "nthread": 1,
    }
    for key in [
        "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_weight",
    ]:
        if key in params:
            train_params[key] = params[key]
    evals = [(dtrain, "train"), (dval, "validation")]
    if early_stopping_rounds is not None:
        callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
        bst = xgb.train(
            train_params, dtrain, num_boost_round=num_boost_round,
            evals=evals, callbacks=callbacks, verbose_eval=False,
        )
        best_iteration = bst.num_boosted_rounds()
    else:
        bst = xgb.train(
            train_params, dtrain, num_boost_round=num_boost_round,
            evals=evals, verbose_eval=False,
        )
        best_iteration = num_boost_round
    sklearn_params = {
        k: v for k, v in params.items()
        if k not in ["eval_metric", "early_stopping_rounds", "n_jobs"]
    }
    sklearn_params["n_estimators"] = best_iteration
    model = xgb.XGBRegressor(**sklearn_params)
    model._Booster = bst
    model.n_estimators = best_iteration
    return model, best_iteration


# ---- 特征工程 ----
def build_features(df, target_col, external_cols=None, seq_len=12, keep_raw=True):
    """基于全部原始外部特征构建衍生特征；keep_raw 为 True 时保留原始外部特征列，
    否则只保留衍生特征。"""
    df_feat = df.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex) and "date" in df_feat.columns:
        df_feat = df_feat.set_index("date")

    # ---- 需求滞后与差分 ----
    for lag in range(1, seq_len + 1):
        df_feat[f"demand_lag_{lag}"] = df_feat[target_col].shift(lag)
    df_feat["demand_diff_1"] = df_feat[target_col].diff(1).shift(1)
    df_feat["demand_diff_12"] = df_feat[target_col].diff(12).shift(1)

    # ---- 需求滚动统计 ----
    for window in [3, 6, 12]:
        df_feat[f"demand_roll_mean_{window}"] = (
            df_feat[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        )
    for window in [3, 6]:
        df_feat[f"demand_roll_std_{window}"] = (
            df_feat[target_col].shift(1).rolling(window=window, min_periods=1).std()
        )

    # ---- 外部特征滞后 ----
    for col in external_cols or []:
        if col in df_feat.columns:
            for lag in [1, 2, 3]:
                df_feat[f"ext_{col}_lag_{lag}"] = df_feat[col].shift(lag)

    # ---- 时间特征 ----
    df_feat["month"] = df_feat.index.month
    df_feat["quarter"] = df_feat.index.quarter
    if "is_spring_festival" in df_feat.columns:
        df_feat["is_spring_festival"] = df_feat["is_spring_festival"].fillna(0).astype(int)

    # ---- 目标列 ----
    df_feat["target"] = df_feat[target_col]
    if target_col in df_feat.columns:
        df_feat = df_feat.drop(columns=[target_col])

    # ---- 列清理 ----
    # 删除非特征列(物资/区域标识等)，但保留原始外部特征(若 keep_raw=True)
    id_cols = ["region", "material_id", "material_name", "material_type", "year"]
    drop_cols = [c for c in id_cols if c in df_feat.columns]
    if not keep_raw:
        # 原逻辑: 删除原始外部特征(保留衍生特征)
        raw_drop_cols = [c for c in (external_cols or []) if c != "is_spring_festival"]
        drop_cols.extend([c for c in raw_drop_cols if c in df_feat.columns])
    df_feat = df_feat.drop(columns=drop_cols, errors="ignore")

    return df_feat


# ---- 数据加载 ----
def load_xgb_data(seq_len=12, keep_raw_features=True):
    """加载数据并使用全部原始外部特征，不依赖特征选择文件。"""
    print("📂 加载原始数据(全部特征版)...")
    df = load_raw_data(SIM_DATASET_PATH)

    # 不再读取 FEATURE_SELECTION_FILE_PATH，直接使用 RAW_EXTERNAL_COLS
    print(f"   原始外部特征: {len(RAW_EXTERNAL_COLS)} 维")
    print(f"   {RAW_EXTERNAL_COLS}")

    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)
    print(f"   训练集: {len(train_clean)} | 验证集: {len(val_clean)} | 测试集: {len(test_clean)}")

    # 跨集特征工程(注意: 使用 all_clean 保证滚动窗口不泄露，但滞后特征需严格时序)
    all_clean = pd.concat([train_clean, val_clean, test_clean])
    all_feat = build_features(
        all_clean, DEMAND_COL, RAW_EXTERNAL_COLS, seq_len, keep_raw=keep_raw_features
    ).dropna()

    # 按索引切分回训练/验证/测试
    train_feat = all_feat[all_feat.index.isin(train_clean.index)]
    val_feat = all_feat[all_feat.index.isin(val_clean.index)]
    test_feat = all_feat[all_feat.index.isin(test_clean.index)]

    print(f"   总特征数(含衍生): {len(train_feat.columns) - 1}")
    print(f"   保留原始特征: {keep_raw_features}")
    return train_feat, val_feat, test_feat


# ---- 训练 ----
def train_xgb(train_df, val_df=None, params=None):
    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train, y_train = train_df[feature_cols].values, train_df["target"].values
    default_params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)
    best_iteration = default_params["n_estimators"]
    if val_df is not None and len(val_df) > 0:
        X_val, y_val = val_df[feature_cols].values, val_df["target"].values
        stage1_model, best_iteration = _train_xgb_core(
            X_train, y_train, X_val, y_val, default_params,
            default_params["n_estimators"], early_stopping_rounds=50,
        )
        print(f"   ✅ 早停完成 | 最佳迭代: {best_iteration}")
    print(f"   🔒 最终训练: 固定 n_estimators={best_iteration}")
    final_model, _ = _train_xgb_core(
        X_train, y_train, X_train, y_train, default_params,
        best_iteration, early_stopping_rounds=None,
    )
    return {
        "model": final_model,
        "params": default_params,
        "best_iteration": best_iteration,
        "feature_cols": feature_cols,
    }


# ---- 评估 ----
def evaluate_xgb(model, test_df, feature_cols):
    X_test, y_true = test_df[feature_cols].values, test_df["target"].values
    y_pred = np.maximum(model.predict(X_test), 0)
    metrics = calculate_metrics(y_true, y_pred, is_normalized=False)
    return {
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "MAPE": metrics["MAPE"],
        "R2": metrics["R2"],
        "y_true": y_true,
        "y_pred": y_pred,
        "n_params": model.n_estimators * len(feature_cols),
    }


# ---- 超参搜索 ----
def search_hyperparameters(train_df, val_df):
    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train, y_train = train_df[feature_cols].values, train_df["target"].values
    X_val, y_val = val_df[feature_cols].values, val_df["target"].values
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 500, 1000],
        "subsample": [0.7, 0.8, 1.0],
    }
    print("\n🔍 开始超参数搜索...")
    best_rmse, best_params = float("inf"), None
    total = np.prod([len(v) for v in param_grid.values()])
    combo_count = 0
    for max_depth, lr, n_est, subsample in product(*param_grid.values()):
        combo_count += 1
        params = {
            "max_depth": max_depth,
            "learning_rate": lr,
            "n_estimators": n_est,
            "subsample": subsample,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "random_state": RANDOM_SEED,
            "tree_method": "hist",
        }
        model, best_iter = _train_xgb_core(
            X_train, y_train, X_val, y_val, params, n_est, early_stopping_rounds=30,
        )
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params.copy()
            best_params["best_iteration"] = best_iter
            print(
                f"   [{combo_count}/{total}] 新最优! RMSE={rmse:.2f} | "
                f"depth={max_depth}, lr={lr}, iter={best_iter}"
            )
    print(f"\n   ✅ 搜索完成 | 最优 RMSE: {best_rmse:.2f}")
    return best_params


# ---- 特征重要性 ----
def analyze_feature_importance(model, feature_cols, top_n=20, save_dir=None):
    if hasattr(model, "_Booster") and model._Booster is not None:
        importance_dict = model._Booster.get_score(importance_type="weight")
        importance = np.array([
            importance_dict.get(f"f{i}", 0.0) for i in range(len(feature_cols))
        ])
    else:
        importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n📊 特征重要性 Top-{min(top_n, len(feat_imp))}:")
    print(feat_imp.head(top_n).to_string(index=False))

    if save_dir:
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)
        plot_df = feat_imp.head(top_n).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(plot_df["feature"], plot_df["importance"], color="#3498db", edgecolor="white")
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"XGBoost Top-{top_n} Feature Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "xgboost_feature_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 特征重要性图已保存: {save_path}")

        csv_path = os.path.join(DATA_FIGURES_DIR, "xgboost_feature_importance.csv")
        os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
        feat_imp.to_csv(csv_path, index=False)
    return feat_imp


# ---- 实验主流程 ----
def run_xgb_experiment(search=False, seq_len=SEQ_LEN, keep_raw_features=True):
    print("\n" + "=" * 70)
    print("🧪 XGBoost 对比实验 (全部原始特征版)")
    print("=" * 70)

    t_start = time.time()
    train_df, val_df, test_df = load_xgb_data(seq_len, keep_raw_features=keep_raw_features)
    train_val_df = pd.concat([train_df, val_df], axis=0)

    if search:
        best_params = search_hyperparameters(train_df, val_df)
        best_iteration = best_params.pop("best_iteration", 500)
    else:
        temp = train_xgb(train_df, val_df)
        best_iteration = temp["best_iteration"]
        best_params = {}

    print(f"\n{'=' * 50}")
    print("🏋️ 最终训练")
    print(f"{'=' * 50}")
    print(f"   合并集: {len(train_val_df)} 样本 | 固定 n_estimators={best_iteration}")
    best_params["n_estimators"] = best_iteration
    train_result = train_xgb(train_val_df, val_df=None, params=best_params)
    model = train_result["model"]
    feature_cols = train_result["feature_cols"]

    print(f"\n{'=' * 50}")
    print("📊 测试集评估")
    print(f"{'=' * 50}")
    eval_results = evaluate_xgb(model, test_df, feature_cols)
    X_test, y_true = test_df[feature_cols].values, test_df["target"].values
    y_pred = np.maximum(model.predict(X_test), 0)
    pred_csv_path = os.path.join(DATA_FIGURES_DIR, "xgboost_predictions.csv")
    pd.DataFrame({
        "true_demand": y_true,
        "pred_demand": y_pred,
    }).to_csv(pred_csv_path, index=False)
    print(f"💾 预测曲线数据已保存: {pred_csv_path}")
    print(f" RMSE:  {eval_results['RMSE']:.4f}")
    print(f" MAE:   {eval_results['MAE']:.4f}")
    print(f" MAPE:  {eval_results['MAPE']:.2f}%")
    print(f" R²:    {eval_results['R2']:.4f}")
    print(f" 参数量(近似): {eval_results['n_params']:,}")

    train_time_min = (time.time() - t_start) / 60.0
    print(f" 训练耗时: {train_time_min:.2f} min")

    print(f"\n 📈 对比主模型 (Full Model, R²=0.8858):")
    r2_diff = eval_results["R2"] - 0.8858
    rmse_diff = eval_results["RMSE"] - 303.26
    print(f"    R²  : {eval_results['R2']:.4f} ({r2_diff:+.4f})")
    print(f"    RMSE: {eval_results['RMSE']:.4f} ({rmse_diff:+.1f})")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    analyze_feature_importance(model, feature_cols, top_n=20, save_dir=FIGURES_DIR)

    results = {
        "name": "xgboost_model",
        "description": f"XGBoost_Model (seq_len={seq_len}, n_features={len(feature_cols)})",
        "RMSE": eval_results["RMSE"],
        "MAE": eval_results["MAE"],
        "MAPE": eval_results["MAPE"],
        "R2": eval_results["R2"],
        "n_params": eval_results["n_params"],
        "train_time_min": round(train_time_min, 2),
        "model_type": "xgboost",
        "data_mode": "all_raw_features",
        "n_features": len(feature_cols),
        "best_iteration": train_result["best_iteration"],
    }
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
    results_df = pd.DataFrame([results])
    csv_path = os.path.join(DATA_FIGURES_DIR, "xgboost_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 结果已保存至: {csv_path}")

    log_path = os.path.join(DATA_FIGURES_DIR, "xgboost_experiment_log.json")
    log_data = {
        "seed": 42,
        "xgboost_version": xgb.__version__,
        "numpy_version": np.__version__,
        "search": search,
        "best_r2": float(results["R2"]),
        "best_rmse": float(results["RMSE"]),
        "n_params": results["n_params"],
        "train_time_min": results["train_time_min"],
        "feature_mode": "all_raw_features",
        "raw_external_cols": RAW_EXTERNAL_COLS,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"📝 实验日志已保存: {log_path}")
    return results_df


# ---- 主入口 ----
if __name__ == "__main__":
    seed_everything(seed=42)
    run_xgb_experiment(search=True, keep_raw_features=True)