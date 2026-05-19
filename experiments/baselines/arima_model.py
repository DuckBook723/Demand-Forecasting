"""
文件位置: experiments/baselines/arima_model.py
作用: ARIMA / SARIMA 对比基线实验（支持手动指定阶数与自动寻阶）

本脚本实现了经典的 ARIMA 及其季节性扩展 SARIMA 模型，用于与深度学习模型（VMD-CAT-BiLSTM）
进行公平的性能对比。所有数据预处理（清洗、时间划分）完全复用项目主流程，确保对比基准一致。

主要功能：
1. 加载原始需求数据（不经过 Min-Max 归一化，保持原始尺度）
   - 复用 split_by_time() 与 clean_data_train()/clean_data_with_stats()
   - 自动合并训练集 + 验证集用于模型训练（ARIMA 不需要验证集调参）

2. ARIMAModel 类
   - 支持非季节性 ARIMA(p,d,q) 和季节性 SARIMA(p,d,q)(P,D,Q,m)
   - 可选自动寻阶（依赖 pmdarima.auto_arima），可手动指定搜索范围
   - 提供滚动预测（rolling forecast）：每步用历史真实值重新拟合模型，模拟真实预测场景
   - 输出 AIC / BIC 信息及残差诊断图（调用 plot_diagnostics）

3. 评估与可视化
   - 使用项目统一的 calculate_metrics() 计算 RMSE, MAE, MAPE, R²（原始尺度）
   - 自动保存预测结果至 CSV（figures/data/arima_results.csv）
   - 绘制真实值 vs 预测值曲线图（保存至 figures/ 目录）

使用方式：
    # 手动指定阶数（默认列表）
    results = run_arima_experiment(mode='manual', order_list=[(1,1,1), (2,1,2)], seasonal=False)

    # 自动寻阶（SARIMA 带季节周期 12）
    results = run_arima_experiment(mode='auto', seasonal=True, m=12, auto_kwargs={'max_p':4, 'max_q':4})

    # 从命令行直接运行（默认自动寻阶，SARIMA，周期12）
    python experiments/baselines/arima_model.py

关键设计原则：
- 严格遵循时间顺序，不泄露未来信息
- 滚动预测使用真实值逐步更新历史，贴近实际业务场景
- 自动寻阶失败时回退到默认参数，保证实验不中断
- 图表和结果保存路径与项目整体风格统一

输出文件：
- figures/data/arima_results.csv      : 汇总指标表
- figures/SARIMA..._plot.png           : 真实-预测对比图（每个模型一张）
- 控制台打印详细训练日志（AIC/BIC、参数摘要）
"""

# ======================== 0. 引入包 =======================
import os
import warnings
import time
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from src.models.trainer import calculate_metrics
from src.data.preprocess import (
    load_raw_data,  # 加载原始数据
    split_by_time,  # 按时间顺序划分数据
    clean_data_train,  # 训练集数据清洗
    clean_data_with_stats,  # 测试集数据清洗
)
from configs.config import (
    SIM_DATASET_PATH,  # 原始数据文件路径
    TRAIN_RATIO,  # 训练集比例
    VAL_RATIO,  # 验证集比例
    DEMAND_COL,  # 目标列名
    DATA_FIGURES_DIR,  # 可视化数据路径
    FIGURES_DIR,  # 可视化路径
)
warnings.filterwarnings("ignore")

# ======================== 1. 加载用于 ARIMA 的数据 =======================
def load_arima_data():
    """
    加载并预处理ARIMA模型所需的数据集

    该函数执行以下操作:
    1. 从配置的路径加载原始需求数据
    2. 按时间顺序将数据划分为训练集、验证集和测试集
    3. 对训练集进行数据清洗，获取清洗统计信息
    4. 使用相同的统计信息对验证集和测试集进行清洗，确保数据一致性
    5. 提取需求列作为时间序列数据
    6. 合并训练集和验证集用于模型训练

    Returns:
        tuple: 包含四个时间序列数据的元组
            - train_series (pd.Series): 训练集需求时间序列
            - val_series (pd.Series): 验证集需求时间序列
            - test_series (pd.Series): 测试集需求时间序列
            - train_val_series (pd.Series): 训练集和验证集合并后的需求时间序列

    Note:
        - 使用训练集的统计信息来清洗验证集和测试集，避免数据泄露
        - 所有数据集保持时间顺序，不进行随机打乱
        - 输出各数据集的样本数量（以月为单位）
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

    train_val_series = pd.concat([train_series, val_series])

    print(f"   训练集: {len(train_series)} 个月")
    print(f"   验证集: {len(val_series)} 个月")
    print(f"   测试集: {len(test_series)} 个月")
    print(f"   训练+验证集: {len(train_val_series)} 个月")

    return train_series, val_series, test_series, train_val_series

# ======================== 2. ARIMA 模型 =======================
class ARIMAModel:
    """
    ARIMA/SARIMA时间序列预测模型封装类

    该类提供了完整的ARIMA模型实现，包括:
    - 数据平稳性检验
    - 自动定阶功能
    - 模型训练与拟合
    - 滚动预测能力

    支持纯ARIMA模型和季节性SARIMA模型
    """

    def __init__(self, order=(1, 1, 1), seasonal=False, m=12, seasonal_order=None):
        """
        初始化ARIMA模型

        Args:
            order (tuple): ARIMA模型的阶数(p, d, q)
                - p: 自回归(AR)项的阶数
                - d: 差分(I)的次数
                - q: 移动平均(MA)项的阶数
                默认值为(1, 1, 1)
            seasonal (bool): 是否使用季节性模型
                - True: 使用SARIMA模型
                - False: 使用纯ARIMA模型
                默认值为False
            m (int): 季节性周期长度
                - 对于月度数据，m=12表示年度季节性
                - 仅在seasonal=True时有效
                默认值为12
            seasonal_order (tuple, optional): 季节性阶数(P, D, Q, m)
                - P: 季节性自回归项的阶数
                - D: 季节性差分次数
                - Q: 季节性移动平均项的阶数
                - m: 季节性周期
                如果为None且seasonal=True，则使用默认值(1, 0, 1, m)

        Attributes:
            self.order: 存储ARIMA模型阶数
            self.seasonal: 存储是否启用季节性
            self.m: 存储季节性周期
            self.fitted_model: 存储拟合后的模型对象，初始为None
            self.seasonal_order: 存储季节性阶数
        """
        super().__init__()
        self.order = order
        self.seasonal = seasonal
        self.m = m
        self.fitted_model = None
        self.seasonal_order = seasonal_order

    def check_stationarity(self, series: pd.Series, alpha=0.05):
        """
        使用ADF检验(Augmented Dickey-Fuller Test)检查时间序列的平稳性

        ADF检验是一种单位根检验方法，用于判断时间序列是否平稳。
        非平稳序列需要进行差分处理才能用于ARIMA建模。

        Args:
            series (pd.Series): 待检验的时间序列数据
            alpha (float): 显著性水平阈值，默认为0.05
                - 当p-value < alpha时，拒绝原假设，认为序列是平稳的
                - 当p-value >= alpha时，无法拒绝原假设，认为序列是非平稳的

        Returns:
            bool: 序列是否平稳
                - True: 序列在给定显著性水平下是平稳的
                - False: 序列在给定显著性水平下是非平稳的

        Note:
            - 会自动移除序列中的缺失值(dropna)
            - 输出ADF检验的p-value和平稳性判断结果
            - 如果序列非平稳，建议增加差分阶数d>0
        """
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < alpha
        print(f"   ADF检验 p-value: {p_value:.4f} -> 序列 {'平稳' if is_stationary else '非平稳，建议差分(d>0)'}")
        return is_stationary

    def auto_select_order(self, train_series: pd.Series, **auto_kwargs):
        """
        使用pmdarima库自动选择最优的ARIMA模型阶数

        该函数基于信息准则(如AIC、BIC)自动搜索最优的(p,d,q)组合。
        支持网格搜索和逐步搜索两种策略，可显著减少手动调参的工作量。

        Args:
            train_series (pd.Series): 用于训练的时间序列数据
            **auto_kwargs: 可选的额外参数，会覆盖默认配置
                常用参数包括:
                - max_p (int): 自回归项的最大阶数，默认5
                - max_q (int): 移动平均项的最大阶数，默认5
                - d (int): 强制指定的差分阶数，None表示自动选择
                - stepwise (bool): 是否使用逐步搜索，True更快，False更精确但更慢
                - information_criterion (str): 信息准则类型，如'aic'或'bic'

        Returns:
            tuple: 包含两个阶数元组的元组
                - order (tuple): 最优的ARIMA阶数(p, d, q)
                - seasonal_order (tuple): 最优的季节性阶数(P, D, Q, m)

        Raises:
            ImportError: 当未安装pmdarima库时抛出
            ValueError: 当auto_arima未能找到合适模型时抛出

        Note:
            - 默认使用逐步搜索(stepwise=True)以提高效率
            - 设置suppress_warnings=True抑制警告信息
            - 开启trace=True显示搜索进度
            - 如果启用季节性，会同时优化季节性阶数
            - 搜索失败时会提供详细的错误提示
        """
        try:
            from pmdarima import auto_arima

            default_kwargs = {
                "start_p": 0, "start_q": 0,
                "max_p": 5, "max_q": 5,
                "d": None,
                "seasonal": self.seasonal,
                "m": self.m if self.seasonal else 1,
                "stepwise": True,
                "suppress_warnings": True,
                "error_action": 'warn',  # 改为 warn，方便看到警告
                "trace": True,  # 自动寻阶时开启 trace 看进度
            }
            default_kwargs.update(auto_kwargs)

            print("   🔍 开始自动寻阶...")
            model = auto_arima(train_series, **default_kwargs)

            # 防止 auto_arima 搜索失败返回 None
            if model is None:
                raise ValueError("auto_arima 未能找到合适的模型，请检查数据或放宽搜索条件。")

            self.order = model.order
            self.seasonal_order = model.seasonal_order if self.seasonal else (0, 0, 0, 0)
            print(f"   ✅ 自动定阶结果: order={self.order}, seasonal_order={self.seasonal_order}")
            return self.order, self.seasonal_order
        except ImportError:
            raise ImportError("自动寻阶需要安装 pmdarima，请运行: pip install pmdarima")

    def fit(self, train_series: pd.Series):
        """
        训练ARIMA/SARIMA模型

        根据配置的阶数和季节性选项，构建并拟合ARIMA或SARIMA模型。
        对于季节性模型，如果未指定季节性阶数，会使用默认值。

        Args:
            train_series (pd.Series): 训练数据的时间序列

        Returns:
            fitted_model: 拟合后的模型对象
                - 包含模型的所有统计信息和诊断指标
                - 可通过summary()方法查看详细报告

        Note:
            - 对于季节性模型，默认季节性阶数为(1, 0, 1, m)
            - 设置enforce_stationarity=False和enforce_invertibility=False以避免收敛问题
            - 输出模型的参数量、AIC和BIC值用于模型评估
            - AIC和BIC越小表示模型越好，但也要考虑过拟合风险
        """
        if self.seasonal and self.seasonal_order is None:
            self.seasonal_order = (1, 0, 1, self.m)
            print(f"   使用默认季节性阶数: {self.seasonal_order}")

        if not self.seasonal:
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

        self.fitted_model = self.model.fit()

        p, d, q = self.order
        P, D, Q, _ = self.seasonal_order
        n_params = p + q + P + Q

        print(f"   ✅ SARIMA{self.order}x{self.seasonal_order} 训练完成 | 参数量: {n_params}")
        print(f"   AIC: {self.fitted_model.aic:.2f} | BIC: {self.fitted_model.bic:.2f}")
        return self.fitted_model

    def rolling_forecast(self, test_series, train_series):
        """
        执行滚动预测(Recursive Forecasting)

        滚动预测是一种动态预测策略，每预测一个时间点就将真实值加入训练历史，
        然后重新训练模型进行下一步预测。这种方法能够利用最新的信息提高预测精度。

        Args:
            test_series (pd.Series): 测试集时间序列，用于逐步预测
            train_series (pd.Series): 训练集时间序列，作为初始训练数据

        Returns:
            np.ndarray: 预测结果数组，长度等于test_series的长度
                - 每个元素对应测试集一个时间点的预测值
                - 按照时间顺序排列

        Note:
            - 采用一步预测策略，每次只预测下一个时间点
            - 尝试使用append方法更新模型以提高效率
            - 如果append失败，则重新拟合整个历史数据
            - 每20步输出一次进度信息
            - 滚动预测比静态预测计算成本更高，但通常更准确
        """
        print(f"🔮 滚动预测测试集 ({len(test_series)} 步)...")

        predictions = []
        if self.seasonal:
            model = SARIMAX(train_series, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = ARIMA(train_series, order=self.order)

        res = model.fit()

        for i, (idx, val) in enumerate(test_series.items()):
            forecast = res.forecast(steps=1)
            pred = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            predictions.append(pred)

            try:
                res = res.append([val])
            except Exception:
                history = pd.concat([train_series, test_series.iloc[:i + 1]])
                if self.seasonal:
                    new_model = SARIMAX(history, order=self.order, seasonal_order=self.seasonal_order)
                else:
                    new_model = ARIMA(history, order=self.order)
                res = new_model.fit()

            if (i + 1) % 20 == 0:
                print(f"   进度: {i + 1}/{len(test_series)}")

        return np.array(predictions)

# ========================= 3. 模型评估 =======================
def evaluate_arima(test_series, train_val_series, order=(1, 1, 1), seasonal=False, m=12, seasonal_order=None):
    """
    评估ARIMA/SARIMA模型的预测性能

    该函数执行完整的模型评估流程:
    1. 创建ARIMA模型实例
    2. 使用滚动预测生成测试集预测值
    3. 计算多个评估指标(RMSE, MAE, MAPE, R²)
    4. 打印模型摘要和诊断图
    5. 返回包含所有评估结果的字典

    Args:
        test_series (pd.Series): 测试集时间序列，用于评估模型泛化能力
        train_val_series (pd.Series): 训练集和验证集的合并数据，用于模型训练
        order (tuple): ARIMA模型阶数(p, d, q)，默认为(1, 1, 1)
        seasonal (bool): 是否使用季节性模型，默认为False
        m (int): 季节性周期长度，默认为12（年度周期性）
        seasonal_order (tuple, optional): 季节性阶数(P, D, Q, m)，默认为None

    Returns:
        dict: 包含评估结果的字典，字段包括:
            - name (str): 模型名称('arima'或'sarima')
            - description (str): 模型描述，包含完整阶数信息
            - RMSE (float): 均方根误差，衡量预测值与真实值的偏差
            - MAE (float): 平均绝对误差，对异常值较鲁棒
            - MAPE (float): 平均绝对百分比误差，便于业务理解
            - R2 (float): 决定系数，衡量模型解释方差的能力
            - n_params (int): 模型参数数量
            - train_time_min (float): 训练时间（分钟），初始为0
            - model_type (str): 模型类型标识
            - data_mode (str): 数据模式标识('raw_demand')
            - order (str): 阶数字符串表示
            - time_index (pd.Index): 测试集的时间索引
            - y_true (np.ndarray): 测试集真实值数组
            - y_pred (np.ndarray): 测试集预测值数组

    Note:
        - 使用滚动预测进行评估，结果更可靠但计算时间较长
        - 会显示完整的模型统计摘要和诊断图
        - 诊断图包括残差分析，用于检验模型假设是否成立
        - 评估指标越小越好（R²越大越好）
    """
    model = ARIMAModel(order=order, seasonal=seasonal, m=m, seasonal_order=seasonal_order)

    y_pred = model.rolling_forecast(test_series, train_val_series)
    y_true = test_series.values

    metrics = calculate_metrics(y_true, y_pred, is_normalized=False)

    p, d, q = order
    P, D, Q, _ = seasonal_order if seasonal else (0, 0, 0, 0)
    n_params = p + q + P + Q

    # 动态生成描述信息
    if seasonal and seasonal_order:
        desc = f'SARIMA{order}x{seasonal_order}'
    else:
        desc = f'ARIMA{order}'

    return {
        'name': 'sarima' if seasonal else 'arima',
        'description': desc,
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'n_params': n_params,
        'train_time_min': 0,
        'model_type': 'sarima' if seasonal else 'arima',
        'data_mode': 'raw_demand',
        'order': str(order),
        'time_index': test_series.index,
        'y_true': y_true,
        'y_pred': y_pred,
    }

# ========================= 4. 模型训练 =======================
def run_arima_experiment(mode='manual', order_list=None, auto_kwargs=None, seasonal=False, m=12):
    """
    运行完整的ARIMA/SARIMA对比实验

    该函数是整个实验的主入口，支持两种运行模式:
    1. manual模式: 手动指定多个ARIMA阶数组合进行对比实验
    2. auto模式: 使用自动寻阶算法找到最优阶数

    实验流程包括:
    - 加载和预处理数据
    - 数据平稳性检验
    - 模型训练和预测（支持多组参数对比）
    - 计算评估指标
    - 保存实验结果到CSV文件
    - 生成预测结果可视化图表

    Args:
        mode (str): 实验模式
            - 'manual': 手动模式，遍历order_list中的所有阶数组合
            - 'auto': 自动模式，使用auto_arima自动寻找最优阶数
            默认为'manual'
        order_list (list, optional): 手动模式下的阶数列表
            - 每个元素是一个(p, d, q)元组
            - 如果为None，使用默认的5组阶数组合
            仅在mode='manual'时有效
        auto_kwargs (dict, optional): 自动寻阶的额外参数
            - 可指定max_p, max_q, d, stepwise等参数
            - 如果为None，使用默认配置
            仅在mode='auto'时有效
        seasonal (bool): 是否使用季节性模型，默认为False
        m (int): 季节性周期长度，默认为12

    Returns:
        pd.DataFrame: 包含所有实验结果的DataFrame
            - 每行代表一次实验的结果
            - 包含模型描述、评估指标、参数量等信息
            - 如果所有实验都失败，返回空DataFrame

    Raises:
        ValueError: 当mode参数不是'manual'或'auto'时抛出

    Note:
        - 结果会保存到DATA_FIGURES_DIR目录下的arima_results.csv
        - 会为每个模型生成预测对比图，显示真实值和预测值
        - 图表文件名会根据模型描述自动生成
        - 自动过滤掉y_true和y_pred列后再保存CSV
        - 如果没有任何成功的实验，会返回空DataFrame而不是抛出异常
    """
    print(f"\n{'=' * 70}")
    print(f"🧪 运行 ARIMA 对比实验 (模式: {mode})")
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

            # 【关键修复】将寻阶得到的 seasonal_order 一并传入评估函数
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
            order_list = [
                (1, 1, 1), (2, 1, 2), (3, 1, 3),
                (0, 1, 1), (1, 1, 0),
            ]

        for order in order_list:
            print(f"\n{'🔹' * 35}")
            print(f" ARIMA{order}")
            print(f"{'🔹' * 35}")
            start_time = time.time()

            try:
                result = evaluate_arima(test_series, train_val_series, order=order, seasonal=seasonal, m=m)
                result['train_time_min'] = round((time.time() - start_time) / 60, 2)
                all_results.append(result)

                print(f"\n   📊 测试集评估结果:")
                print(f"      RMSE:  {result['RMSE']:.4f}")
                print(f"      MAE:   {result['MAE']:.4f}")
                print(f"      MAPE:  {result['MAPE']:.2f}%")
                print(f"      R2:    {result['R2']:.4f}")

            except Exception as e:
                print(f"❌ ARIMA{order} 实验失败: {e}")
    else:
        raise ValueError("mode 必须为 'manual' 或 'auto'")

    # 【关键修复】保护空结果集，防止 KeyError
    if not all_results:
        print("\n⚠️ 没有成功运行的实验，无结果汇总。")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(DATA_FIGURES_DIR, 'arima_results.csv')
    results_df.drop(columns=['y_true', 'y_pred'], errors='ignore').to_csv(save_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"📋 ARIMA 实验汇总结果")
    print(f"{'=' * 70}")
    print(results_df[['description', 'RMSE', 'MAE', 'MAPE', 'R2', 'n_params']].to_string(index=False))
    print(f"\n💾 结果已保存至: {save_path}")

    return results_df

# =========================== 测试 ==========================
if __name__ == '__main__':
    custom_auto_params = {
        'max_p': 4,
        'max_q': 4,
        'd': 1,  # 强制差分阶数为 1
        'stepwise': False,  # 使用网格搜索（更慢但可能更优）
        'information_criterion': 'bic'  # 使用 BIC 作为评判标准
    }
    results = run_arima_experiment(mode='auto', auto_kwargs=custom_auto_params, seasonal=True, m=12)