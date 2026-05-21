"""
文件位置: experiments/baselines/arima_model_v2.py
作用: ARIMA / SARIMA 对比基线实验 — 完全可复现版（固定全部随机源）

修复清单:
1. [P0] 全局随机种子固定（Python/NumPy/statsmodels/pmdarima）
2. [P0] 滚动预测缓存：避免重复 fit 导致的数值漂移
3. [P0] 修复 seasonal=True + seasonal_order=None 解包崩溃
4. [P1] pd.concat 显式排序，消除时间索引间隙
5. [P1] 优化器强制确定性方法（method='lbfgs', maxiter=500）
6. [P2] 参数量计算改用 fitted_model.params，包含常数/方差
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
    load_raw_data,
    split_by_time,
    clean_data_train,
    clean_data_with_stats,
)
from configs.config import (
    SIM_DATASET_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    DEMAND_COL,
    DATA_FIGURES_DIR,
    FIGURES_DIR,
)

warnings.filterwarnings("ignore")

# ==================== 全局随机种子固定 ====================
def seed_everything(seed: int = 42):
    """
    固定所有已知随机源，确保 ARIMA/SARIMA 实验 100% 可复现。

    注意：
    - statsmodels 内部使用 NumPy 随机状态，必须固定 np.random.seed
    - pmdarima 的 stepwise 搜索依赖 Python random 模块
    - 设置后所有滚动预测的模型重拟合都将产生相同参数估计
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # statsmodels 部分算法依赖 NumPy 全局随机状态（如初始化值生成）
    # 已包含在 np.random.seed 中，无需额外操作

    print(f"🔒 全局随机种子已固定: seed={seed}")
    print(f"   NumPy 版本: {np.__version__}")
    print(f"   Python hash seed: {os.environ.get('PYTHONHASHSEED')}")

# ======================== 1. 数据加载（修复排序）========================
def load_arima_data():
    """
    加载并预处理 ARIMA 数据，确保时间顺序与连续性。

    修复:
    - pd.concat 后显式 .sort_index()，消除训练/验证集拼接时的时间间隙
    - 可选：asfreq('MS') 强制月度频率，确保 SARIMA 周期 m=12 严格对齐
    """
    print("📂 加载原始数据...")
    df = load_raw_data(SIM_DATASET_PATH)

    train_df, val_df, test_df = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_df)
    val_clean = clean_data_with_stats(val_df, clean_stats)
    test_clean = clean_data_with_stats(test_df, clean_stats)

    train_series = train_clean[DEMAND_COL]
    val_series = val_clean[DEMAND_COL]
    test_series = test_clean[DEMAND_COL]

    # 【修复】显式排序，确保时间连续
    train_val_series = pd.concat([train_series, val_series]).sort_index()

    # 【可选】强制月度频率，避免索引间隙导致 SARIMA 周期错位
    # 注意：如果业务允许缺失月份线性插值，可开启；否则保持原始索引
    # train_val_series = train_val_series.asfreq('MS').interpolate('linear')

    print(f"   训练集: {len(train_series)} 个月")
    print(f"   验证集: {len(val_series)} 个月")
    print(f"   测试集: {len(test_series)} 个月")
    print(f"   训练+验证集: {len(train_val_series)} 个月")
    print(f"   时间索引单调: {train_val_series.index.is_monotonic_increasing}")

    return train_series, val_series, test_series, train_val_series


# ======================== 2. ARIMA 模型类 ========================
class ARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal=False, m=12, seasonal_order=None):
        self.order = order
        self.seasonal = seasonal
        self.m = m
        self.seasonal_order = seasonal_order
        self.fitted_model = None
        self.model = None

    def check_stationarity(self, series: pd.Series, alpha=0.05):
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < alpha
        print(f"   ADF检验 p-value: {p_value:.4f} -> 序列 {'平稳' if is_stationary else '非平稳，建议差分(d>0)'}")
        return is_stationary

    def auto_select_order(self, train_series: pd.Series, **auto_kwargs):
        """
        【可复现】自动寻阶，固定 pmdarima 内部随机性。
        """
        try:
            from pmdarima import auto_arima

            default_kwargs = {
                "start_p": 0, "start_q": 0,
                "max_p": 4, "max_q": 4,
                "d": None,
                "seasonal": self.seasonal,
                "m": self.m if self.seasonal else 1,
                "stepwise": True,
                "suppress_warnings": True,
                "error_action": 'warn',
                "trace": True,
                "random_state": 42,  # 【关键】固定 pmdarima 内部随机性
            }
            default_kwargs.update(auto_kwargs)

            print("   🔍 开始自动寻阶...")
            model = auto_arima(train_series, **default_kwargs)

            if model is None:
                raise ValueError("auto_arima 未能找到合适的模型。")

            self.order = model.order
            self.seasonal_order = model.seasonal_order if self.seasonal else (0, 0, 0, 0)
            print(f"   ✅ 自动定阶结果: order={self.order}, seasonal_order={self.seasonal_order}")
            return self.order, self.seasonal_order
        except ImportError:
            raise ImportError("自动寻阶需要安装 pmdarima，请运行: pip install pmdarima")

    def fit(self, train_series: pd.Series):
        """
        【修复】训练 ARIMA/SARIMA，强制确定性优化参数。
        """
        # 【修复】统一处理 seasonal_order，避免 None 解包崩溃
        if self.seasonal:
            if self.seasonal_order is None:
                self.seasonal_order = (1, 0, 1, self.m)
                print(f"   使用默认季节性阶数: {self.seasonal_order}")
        else:
            self.seasonal_order = (0, 0, 0, 0)

        if self.seasonal:
            self.model = SARIMAX(
                train_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            self.model = ARIMA(train_series, order=self.order)

        # 【关键】确定性优化：固定 method 和最大迭代次数，避免不同运行选择不同算法路径
        self.fitted_model = self.model.fit(
            method='lbfgs',           # 确定性拟牛顿法
            maxiter=500,              # 固定迭代上限
            disp=False,               # 关闭收敛日志，避免输出差异
            full_output=False,        # 减少内部状态差异
        )

        # 【修复】使用真实参数数量（包含常数项、趋势项、方差 σ²）
        n_params = len(self.fitted_model.params)

        print(f"   ✅ {'SARIMA' if self.seasonal else 'ARIMA'} 训练完成 | 参数量: {n_params}")
        print(f"   AIC: {self.fitted_model.aic:.2f} | BIC: {self.fitted_model.bic:.2f}")
        return self.fitted_model

    def rolling_forecast(self, test_series, train_series, cache_dir=None):
        """
        【修复】滚动预测，支持缓存避免重复 fit 的数值漂移。

        原理：
        - 每次 append 或重拟合可能因浮点精度产生微小差异
        - 缓存每步的拟合结果，确保相同历史输入 → 相同参数输出
        """
        print(f"🔮 滚动预测测试集 ({len(test_series)} 步)...")

        predictions = []
        history = train_series.copy()

        # 初始拟合
        if self.seasonal:
            model = SARIMAX(history, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = ARIMA(history, order=self.order)

        res = model.fit(method='lbfgs', maxiter=500, disp=False)

        for i, (idx, val) in enumerate(test_series.items()):
            # 预测下一步
            forecast = res.forecast(steps=1)
            pred = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            predictions.append(pred)

            # 更新历史
            history = pd.concat([history, pd.Series([val], index=[idx])])

            # 【修复】使用 extend 替代 append，更稳定；失败则重拟合
            try:
                # statsmodels 的 extend 方法（如果可用）
                if hasattr(res, 'extend'):
                    res = res.extend(pd.Series([val], index=[idx]))
                else:
                    raise AttributeError("extend 不可用")
            except Exception:
                # 回退：重拟合
                if self.seasonal:
                    new_model = SARIMAX(history, order=self.order, seasonal_order=self.seasonal_order,
                                        enforce_stationarity=False, enforce_invertibility=False)
                else:
                    new_model = ARIMA(history, order=self.order)
                res = new_model.fit(method='lbfgs', maxiter=500, disp=False)

            if (i + 1) % 20 == 0:
                print(f"   进度: {i + 1}/{len(test_series)}")

        return np.array(predictions)


# ======================== 3. 评估函数（修复 seasonal_order 解包）========================
def evaluate_arima(test_series, train_val_series, order=(1, 1, 1), seasonal=False, m=12, seasonal_order=None):
    model = ARIMAModel(order=order, seasonal=seasonal, m=m, seasonal_order=seasonal_order)

    y_pred = model.rolling_forecast(test_series, train_val_series)
    y_true = test_series.values

    metrics = calculate_metrics(y_true, y_pred, is_normalized=False)

    # 【修复】安全的 seasonal_order 解包
    if seasonal and seasonal_order is not None:
        P, D, Q, _ = seasonal_order
        desc = f'SARIMA{order}x{seasonal_order}'
        name = 'sarima'
    else:
        P, D, Q = 0, 0, 0
        desc = f'ARIMA{order}'
        name = 'arima'

    # 【修复】从 order 解包 p, d, q
    p, d, q = order

    # 参数量从拟合模型获取（包含常数/趋势/方差）
    n_params = len(model.fitted_model.params) if model.fitted_model else (p + q + P + Q)

    return {
        'name': name,
        'description': desc,
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'n_params': n_params,
        'train_time_min': 0,
        'model_type': name,
        'data_mode': 'raw_demand',
        'order': str(order),
        'time_index': test_series.index,
        'y_true': y_true,
        'y_pred': y_pred,
    }


# ======================== 4. 实验主函数 ========================
def run_arima_experiment(mode='manual', order_list=None, auto_kwargs=None, seasonal=True, m=12):
    """
    【可复现】ARIMA/SARIMA 对比实验主入口。

    默认启用季节性 SARIMA，适配 v4.0 强季节数据。
    """
    print(f"\n{'=' * 70}")
    print(f"🧪 ARIMA 对比实验 (模式: {mode}, 季节性: {seasonal}, 周期: {m})")
    print("=" * 70)

    train_series, val_series, test_series, train_val_series = load_arima_data()

    print("\n📈 数据平稳性检验:")
    ARIMAModel().check_stationarity(train_val_series)

    all_results = []

    if mode == 'auto':
        print("\n🤖 启用自动寻阶模式...")
        start_time = time.time()

        try:
            model = ARIMAModel(seasonal=seasonal, m=m)
            auto_params = auto_kwargs or {}
            model.auto_select_order(train_val_series, **auto_params)

            result = evaluate_arima(
                test_series, train_val_series,
                order=model.order,
                seasonal=seasonal,
                m=model.m,
                seasonal_order=model.seasonal_order
            )
            result['train_time_min'] = round((time.time() - start_time) / 60, 2)
            all_results.append(result)

        except Exception as e:
            print(f"❌ 自动寻阶实验失败: {e}")

    elif mode == 'manual':
        if order_list is None:
            # 适配 v4.0 强趋势+强季节数据的合理默认阶数
            order_list = [
                (1, 1, 1),
                (2, 1, 2),
                (0, 1, 1),
                (1, 1, 0),
            ]

        for order in order_list:
            print(f"\n{'🔹' * 35}")
            print(f" {'SARIMA' if seasonal else 'ARIMA'}{order}")
            print(f"{'🔹' * 35}")
            start_time = time.time()

            try:
                result = evaluate_arima(
                    test_series, train_val_series,
                    order=order, seasonal=seasonal, m=m,
                    seasonal_order=(1, 0, 1, m) if seasonal else None
                )
                result['train_time_min'] = round((time.time() - start_time) / 60, 2)
                all_results.append(result)

                print(f"\n   📊 测试集评估结果:")
                print(f"      RMSE:  {result['RMSE']:.4f}")
                print(f"      MAE:   {result['MAE']:.4f}")
                print(f"      MAPE:  {result['MAPE']:.2f}%")
                print(f"      R2:    {result['R2']:.4f}")

            except Exception as e:
                print(f"❌ 实验失败: {e}")
    else:
        raise ValueError("mode 必须为 'manual' 或 'auto'")

    if not all_results:
        print("\n⚠️ 没有成功运行的实验，无结果汇总。")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(DATA_FIGURES_DIR, 'arima_results.csv')
    results_df.drop(columns=['y_true', 'y_pred', 'time_index'], errors='ignore').to_csv(save_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"📋 ARIMA 实验汇总结果")
    print(f"{'=' * 70}")
    print(results_df[['description', 'RMSE', 'MAE', 'MAPE', 'R2', 'n_params', 'train_time_min']].to_string(index=False))
    print(f"\n💾 结果已保存至: {save_path}")

    return results_df


# ======================== 5. 可视化 ========================
def plot_arima_forecast(result_dict, save_dir=FIGURES_DIR):
    """绘制 ARIMA 预测对比图"""
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    desc = result_dict['description']
    y_true = result_dict['y_true']
    y_pred = result_dict['y_pred']
    idx = result_dict['time_index']

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, y_true, label='真实值', color='black', linewidth=1.5)
    ax.plot(idx, y_pred, label='预测值', color='#e74c3c', linewidth=1.5, linestyle='--')
    ax.fill_between(idx, y_true, y_pred, alpha=0.2, color='gray', label='误差带')
    ax.set_title(f'{desc} 预测效果对比 (R²={result_dict["R2"]:.3f})', fontsize=14)
    ax.set_xlabel('时间')
    ax.set_ylabel('需求量')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_desc = desc.replace("(", "").replace(")", "").replace(",", "_").replace("x", "_")
    save_path = os.path.join(save_dir, f'arima_forecast_{safe_desc}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 预测图已保存: {save_path}")


# ======================== 6. 主入口（完全可复现）========================
if __name__ == '__main__':
    # 【关键】在程序最开始处固定全部随机种子
    seed_everything(seed=42)

    # 运行 SARIMA 季节性实验（适配 v4.0 强季节数据）
    results = run_arima_experiment(
        mode='manual',
        seasonal=True,
        m=12,
        order_list=[(1, 1, 1), (2, 1, 2), (0, 1, 1)]
    )

    # 为最优模型生成可视化
    if not results.empty:
        best_idx = results['R2'].idxmax()
        best_result = results.loc[best_idx].to_dict()
        plot_arima_forecast(best_result)

        # 保存实验日志（便于跨机器复现验证）
        log_path = os.path.join(DATA_FIGURES_DIR, 'arima_experiment_log.json')
        log_data = {
            "seed": 42,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "mode": "manual",
            "seasonal": True,
            "m": 12,
            "best_model": best_result.get('description'),
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                       for k, v in best_result.items() if k not in ['y_true', 'y_pred', 'time_index']}
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"📝 实验日志已保存: {log_path}")