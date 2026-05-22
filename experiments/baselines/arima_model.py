"""
文件位置: experiments/baselines/arima_model.py
作用: ARIMA / SARIMA 对比基线实验 — 输出格式与消融实验对齐版
"""

import os
import sys
import random
import warnings
import time
import json

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.models.trainer import calculate_metrics
from src.data.preprocess import (
    load_raw_data, split_by_time, clean_data_train, clean_data_with_stats,
)
from configs.config import (
    SIM_DATASET_PATH, TRAIN_RATIO, VAL_RATIO, DEMAND_COL,
    DATA_FIGURES_DIR, FIGURES_DIR,
)

warnings.filterwarnings("ignore")


# ==================== 全局随机种子固定 ====================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    print(f"🔒 全局随机种子已固定: seed={seed}")
    print(f"   NumPy 版本: {np.__version__}")
    print(f"   Python hash seed: {os.environ.get('PYTHONHASHSEED')}")


# ======================== 1. 数据加载 ========================
def load_arima_data():
    print("📂 加载原始数据...")
    df = load_raw_data(SIM_DATASET_PATH)
    train_df, val_df, test_df = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_df)
    val_clean = clean_data_with_stats(val_df, clean_stats)
    test_clean = clean_data_with_stats(test_df, clean_stats)

    train_series = train_clean[DEMAND_COL]
    val_series = val_clean[DEMAND_COL]
    test_series = test_clean[DEMAND_COL]
    train_val_series = pd.concat([train_series, val_series]).sort_index()

    print(f"   训练集: {len(train_series)} 个月")
    print(f"   验证集: {len(val_series)} 个月")
    print(f"   测试集: {len(test_series)} 个月")
    return train_series, val_series, test_series, train_val_series


# ======================== 2. ARIMA 模型类 ========================
class ARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal=False, m=12, seasonal_order=None):
        self.order = order
        self.seasonal = seasonal
        self.m = m
        self.seasonal_order = seasonal_order
        self.fitted_model = None

    def check_stationarity(self, series: pd.Series, alpha=0.05):
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < alpha
        print(f"   ADF检验 p-value: {p_value:.4f} -> 序列 {'平稳' if is_stationary else '非平稳，建议差分(d>0)'}")
        return is_stationary

    def fit(self, train_series: pd.Series):
        if self.seasonal and self.seasonal_order is None:
            self.seasonal_order = (1, 0, 1, self.m)

        if self.seasonal:
            self.model = SARIMAX(
                train_series, order=self.order, seasonal_order=self.seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
        else:
            self.model = ARIMA(train_series, order=self.order)

        self.fitted_model = self.model.fit(method='lbfgs', maxiter=500, disp=False)
        n_params = len(self.fitted_model.params)
        print(f"   ✅ {'SARIMA' if self.seasonal else 'ARIMA'} 训练完成 | 参数量: {n_params}")
        print(f"   AIC: {self.fitted_model.aic:.2f} | BIC: {self.fitted_model.bic:.2f}")
        return self.fitted_model

    def rolling_forecast(self, test_series, train_series):
        print(f"🔮 滚动预测测试集 ({len(test_series)} 步)...")
        predictions = []
        history = train_series.copy()

        if self.seasonal:
            model = SARIMAX(history, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = ARIMA(history, order=self.order)
        res = model.fit(method='lbfgs', maxiter=500, disp=False)

        for i, (idx, val) in enumerate(test_series.items()):
            forecast = res.forecast(steps=1)
            pred = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            predictions.append(pred)
            history = pd.concat([history, pd.Series([val], index=[idx])])
            try:
                if hasattr(res, 'extend'):
                    res = res.extend(pd.Series([val], index=[idx]))
                else:
                    raise AttributeError
            except Exception:
                if self.seasonal:
                    new_model = SARIMAX(history, order=self.order, seasonal_order=self.seasonal_order,
                                        enforce_stationarity=False, enforce_invertibility=False)
                else:
                    new_model = ARIMA(history, order=self.order)
                res = new_model.fit(method='lbfgs', maxiter=500, disp=False)
            if (i + 1) % 20 == 0:
                print(f"   进度: {i + 1}/{len(test_series)}")
        return np.array(predictions)


# ======================== 3. 评估函数 ========================
def evaluate_arima(test_series, train_val_series, order=(1, 1, 1), seasonal=False, m=12, seasonal_order=None):
    model = ARIMAModel(order=order, seasonal=seasonal, m=m, seasonal_order=seasonal_order)
    y_pred = model.rolling_forecast(test_series, train_val_series)
    y_true = test_series.values
    metrics = calculate_metrics(y_true, y_pred, is_normalized=False)

    if seasonal and seasonal_order is not None:
        P, D, Q, _ = seasonal_order
        desc = f'SARIMA{order}x{seasonal_order}'
        name = 'sarima'
    else:
        P, D, Q = 0, 0, 0
        desc = f'ARIMA{order}'
        name = 'arima'
    p, d, q = order
    n_params = len(model.fitted_model.params) if model.fitted_model else (p + q + P + Q)

    return {
        'name': name, 'description': desc,
        'RMSE': metrics['RMSE'], 'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'], 'R2': metrics['R2'],
        'n_params': n_params, 'train_time_min': 0,
        'model_type': name, 'data_mode': 'raw_demand',
        'order': str(order), 'time_index': test_series.index,
        'y_true': y_true, 'y_pred': y_pred,
    }


# ======================== 4. 实验主函数 ========================
def run_arima_experiment(mode='manual', order_list=None, seasonal=True, m=12):
    print(f"\n{'=' * 70}")
    print(f"🧪 ARIMA 对比实验 (模式: {mode}, 季节性: {seasonal}, 周期: {m})")
    print(f"{'=' * 70}")

    train_series, val_series, test_series, train_val_series = load_arima_data()

    print("\n📈 数据平稳性检验:")
    ARIMAModel().check_stationarity(train_val_series)

    all_results = []
    if mode == 'manual':
        if order_list is None:
            order_list = [(1, 1, 1), (2, 1, 2), (0, 1, 1)]
        for order in order_list:
            print(f"\n{'=' * 50}")
            print(f"🔹 {'SARIMA' if seasonal else 'ARIMA'}{order}")
            print(f"{'=' * 50}")
            t_start = time.time()
            try:
                result = evaluate_arima(
                    test_series, train_val_series, order=order,
                    seasonal=seasonal, m=m,
                    seasonal_order=(1, 0, 1, m) if seasonal else None
                )
                result['train_time_min'] = round((time.time() - t_start) / 60, 2)
                all_results.append(result)
                print(f"\n   📊 测试集评估结果:")
                print(f"      RMSE:  {result['RMSE']:.4f}")
                print(f"      MAE:   {result['MAE']:.4f}")
                print(f"      MAPE:  {result['MAPE']:.2f}%")
                print(f"      R²:    {result['R2']:.4f}")
            except Exception as e:
                print(f"❌ 实验失败: {e}")

    if not all_results:
        return pd.DataFrame()

    # 汇总输出（与消融实验对齐）
    best = max(all_results, key=lambda x: x['R2'])
    print(f"\n{'=' * 50}")
    print("📊 ARIMA 最优测试集结果")
    print(f"{'=' * 50}")
    print(f" RMSE: {best['RMSE']:.4f}")
    print(f" MAE:  {best['MAE']:.4f}")
    print(f" MAPE: {best['MAPE']:.2f}%")
    print(f" R²:   {best['R2']:.4f}")
    print(f" 参数量: {best['n_params']}")
    print(f" 训练耗时: {best['train_time_min']:.2f} min")

    print(f"\n 📈 对比主模型 (Full Model, R²=0.8858):")
    r2_diff = best['R2'] - 0.8858
    rmse_diff = best['RMSE'] - 303.26
    print(f"    R²  : {best['R2']:.4f} ({r2_diff:+.4f})")
    print(f"    RMSE: {best['RMSE']:.4f} ({rmse_diff:+.1f})")

    # 保存 CSV
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(DATA_FIGURES_DIR, 'arima_results.csv')
    results_df.drop(columns=['y_true', 'y_pred', 'time_index'], errors='ignore').to_csv(csv_path, index=False)
    print(f"\n💾 结果已保存至: {csv_path}")

    # 保存 JSON
    log_path = os.path.join(DATA_FIGURES_DIR, 'arima_experiment_log.json')
    log_data = {
        "seed": 42, "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "mode": mode, "seasonal": seasonal, "m": m,
        "best_model": best['description'],
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                   for k, v in best.items() if k not in ['y_true', 'y_pred', 'time_index']}
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"📝 实验日志已保存: {log_path}")
    return results_df


# ======================== 5. 可视化 ========================
def plot_arima_forecast(result_dict, save_dir=FIGURES_DIR):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    desc = result_dict['description']
    y_true, y_pred, idx = result_dict['y_true'], result_dict['y_pred'], result_dict['time_index']
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, y_true, label='真实值', color='black', linewidth=1.5)
    ax.plot(idx, y_pred, label='预测值', color='#e74c3c', linewidth=1.5, linestyle='--')
    ax.fill_between(idx, y_true, y_pred, alpha=0.2, color='gray', label='误差带')
    ax.set_title(f'{desc} 预测对比 (R²={result_dict["R2"]:.3f})', fontsize=14)
    ax.set_xlabel('时间'); ax.set_ylabel('需求量'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_desc = desc.replace("(", "").replace(")", "").replace(",", "_").replace("x", "_")
    save_path = os.path.join(save_dir, f'arima_forecast_{safe_desc}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"📈 预测图已保存: {save_path}")


# ======================== 6. 主入口 ========================
if __name__ == '__main__':
    seed_everything(seed=42)
    results = run_arima_experiment(mode='manual', seasonal=True, m=12, order_list=[(1, 1, 1), (2, 1, 2), (0, 1, 1)])
    if not results.empty:
        best_idx = results['R2'].idxmax()
        plot_arima_forecast(results.loc[best_idx].to_dict())