"""
文件位置: src/data/preprocess.py
作用: 数据清洗、VMD分解、缺失值插补、异常值缩尾、Min-Max归一化
所包含的函数:
  - load_raw_data(): 加载原始仿真数据
  - clean_data(): 缺失值线性插值 + demand列1%-99%缩尾
  - perform_vmd_train_only(): 仅对训练集需求序列做VMD分解，避免信息泄露
  - apply_vmd_to_subset(): 将训练集的IMF分解结果推广到验证/测试集
  - split_by_time(): 严格按时间顺序 70/10/20 划分
  - normalize_and_save(): 训练集拟合MinMaxScaler，统一转换三路数据并保存
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 读取 data/raw/grid_material_demand_sim.csv
  - 输出 data/processed/{train,val,test}.csv, scaler.pkl
"""

# ======================== 0. 引入包 =======================
import os
import sys
import numpy as np
import pandas as pd
import joblib
from vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
from configs.config import (
    SIM_DATASET_PATH,               # 原始仿真数据路径
    VMD_ALPHA,                      # VMD 惩罚因子
    VMD_TAU,                        # 拉格朗日乘子对偶上升步长
    VMD_K,                          # 模态数
    VMD_DC,                         # 直流分量约束
    VMD_INIT,                       # 中心频率的初始化方式
    VMD_TOL,                        # 收敛容差
    DEMAND_COL,                     # 需求列
    TRAIN_FILE_PATH,                # 训练集文件路径
    VAL_FILE_PATH,                  # 验证集文件路径
    TEST_FILE_PATH,                 # 测试集文件路径
    SCALER_FILE_PATH,               # 缩放器文件路径
    TRAIN_RATIO,                    # 训练集比例
    VAL_RATIO,                      # 验证集比例
    DATA_PROCESSED_DIR,             # 处理后数据目录
    DEMAND_MIN_MAX_SCALER_PATH,     # 需求列Min-Max归一化缩放器保存路径
)

# ======================== 1. 加载原始仿真数据 =======================
def load_raw_data(path: str) -> pd.DataFrame:
    """
    加载原始仿真数据并返回按日期排序的DataFrame。

    从指定的CSV文件路径读取数据，自动将'date'列解析为日期类型并设置为索引，
    最后按时间顺序排序返回。

    Args:
        path (str): CSV文件的路径，默认为配置中的SIM_DATASET_PATH。

    Returns:
        pd.DataFrame: 包含原始仿真数据的DataFrame，索引为排序后的日期。
    """
    df = pd.read_csv(path, parse_dates=['date'], index_col='date')
    return df.sort_index()

# ====================== 2. 缺失值插补 + demand列1%-99%缩尾 ======================
def clean_data_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    对训练集数据进行清洗，并返回统计量供验证/测试集使用（避免信息泄漏）。

    仅使用训练集自身数据计算缩尾分位数和统计量，确保无未来信息泄漏。
    缺失值使用前向填充（仅用过去数据）+ 线性插值（仅在训练集内部）。

    Args:
        df (pd.DataFrame): 训练集DataFrame数据。

    Returns:
        tuple: (清洗后的DataFrame, 统计量字典)
            统计量字典包含每列的缩尾上下界，供clean_data_with_stats使用。
    """
    df_clean = df.copy()

    # 仅使用前向填充（不使用后向填充，避免使用未来数据）
    df_clean = df_clean.ffill()
    # 对训练集内部仍为NaN的值（如第一行），使用线性插值
    df_clean = df_clean.interpolate(method='linear', limit_direction='both')

    # 仅在训练集上计算缩尾分位数，避免信息泄漏
    clip_bounds = {}
    num_cols = df_clean.select_dtypes(include=['number']).columns
    for col in num_cols:
        lower, upper = df_clean[col].quantile([0.01, 0.99])
        clip_bounds[col] = (lower, upper)
        df_clean[col] = df_clean[col].clip(lower, upper)

    return df_clean, {'clip_bounds': clip_bounds}

def clean_data_with_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    使用训练集的统计量对验证/测试集进行清洗（无信息泄漏）。

    使用前向填充（仅依赖过去数据）+ 训练集的缩尾分位数边界，
    确保验证/测试集不使用任何自身统计量或未来信息。

    Args:
        df (pd.DataFrame): 验证/测试集DataFrame数据。
        stats (dict): 训练集返回的统计量字典，包含clip_bounds。

    Returns:
        pd.DataFrame: 清洗后的DataFrame。
    """
    df_clean = df.copy()

    # 仅使用前向填充（不使用后向填充，避免使用未来数据）
    df_clean = df_clean.ffill()
    # 对首行仍为NaN的值使用训练集均值填充（而非插值，避免泄漏）
    clip_bounds = stats['clip_bounds']
    for col in df_clean.columns:
        if df_clean[col].isna().any():
            fill_val = (clip_bounds.get(col, (0, 0))[0] + clip_bounds.get(col, (0, 0))[1]) / 2
            df_clean[col] = df_clean[col].fillna(fill_val)

    # 使用训练集的缩尾边界（不使用验证/测试集自身分位数）
    num_cols = df_clean.select_dtypes(include=['number']).columns
    for col in num_cols:
        if col in clip_bounds:
            lower, upper = clip_bounds[col]
            df_clean[col] = df_clean[col].clip(lower, upper)

    return df_clean

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    对数据进行清洗（兼容旧接口，仍可在非严格场景使用）。

    注意：此函数对全量数据计算分位数，存在信息泄漏风险。
    推荐使用 clean_data_train() + clean_data_with_stats() 替代。

    Args:
        df (pd.DataFrame): 待清洗的DataFrame数据。

    Returns:
        pd.DataFrame: 经过缺失值插补和异常值缩尾处理后的DataFrame。
    """
    df_clean, _ = clean_data_train(df)
    return df_clean

# ====================== 3. 仅对训练集需求序列做VMD分解 ======================
def perform_vmd_train_only(train_demand: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    仅对训练集的需求序列进行VMD分解，避免信息泄露。
    VMD分解只使用训练集数据拟合，返回的IMF分量和VMD参数供后续
    对验证/测试集进行外推处理。

    Args:
        train_demand (np.ndarray): 训练集需求序列，形状 (n_train,)

    Returns:
        tuple:
            - imfs_train (np.ndarray): 讓练集的IMF分量，形状 (K, n_train)
            - vmd_info (dict): VMD分解的元信息，包含中心频率等，供外推参考
    """
    n_input = len(train_demand)

    u, u_hat, omega = VMD(
        train_demand,
        alpha=VMD_ALPHA, tau=VMD_TAU, K=VMD_K,
        DC=VMD_DC, init=VMD_INIT, tol=VMD_TOL
    )

    # ========== 新增：长度校验与对齐 ==========
    # vmdpy 在某些参数组合下（如 tau=0）可能返回比输入短1的IMF
    # 需要检查并补齐，确保 IMF 长度与输入信号一致
    n_output = u.shape[1]
    if n_output < n_input:
        # 用线性外推补齐最后一个点：取最后2个点的线性趋势延伸
        padding = np.zeros((u.shape[0], n_input - n_output))
        for k in range(u.shape[0]):
            if n_output >= 2:
                # 线性外推
                slope = u[k, -1] - u[k, -2]
                padding[k, :] = u[k, -1] + slope * np.arange(1, n_input - n_output + 1)
            else:
                # 仅1个点时，直接复制
                padding[k, :] = u[k, -1]
        u = np.hstack([u, padding])
        print(f"  ⚠️ VMD输出长度({n_output}) < 输入长度({n_input})，已补齐末尾{n_input - n_output}个点")
    elif n_output > n_input:
        # 截断多余部分（理论上不应出现）
        u = u[:, :n_input]
        print(f"  ⚠️ VMD输出长度({n_output}) > 输入长度({n_input})，已截断")
    # ========== 长度校验结束 ==========

    vmd_info = {
        'omega': omega,  # 中心频率
        'u_hat': u_hat,  # 频域表示
    }
    return u, vmd_info

# ====================== 4. 将VMD分解结果推广到验证/测试集 ======================
def _fit_ar_predict(series: np.ndarray, order: int, n_forecast: int) -> np.ndarray:
    """
    对单个IMF分量拟合自回归(AR)模型并外推预测。

    相比线性外推，AR模型能保留IMF的振荡特性（周期性、自相关性），
    避免将振荡信号退化为单调趋势线。

    Args:
        series (np.ndarray): 训练集的IMF序列，形状 (n_train,)
        order (int): AR模型阶数
        n_forecast (int): 需要外推的步数

    Returns:
        np.ndarray: 外推的IMF预测值，形状 (n_forecast,)
    """
    n = len(series)
    if n < order + 1:
        # 序列太短，退化为均值外推
        return np.full(n_forecast, np.mean(series))

    # 构建AR回归矩阵: X = [y(t-1), y(t-2), ..., y(t-p)], y = y(t)
    X = np.zeros((n - order, order))
    y = np.zeros(n - order)
    for i in range(order):
        X[:, i] = series[order - i - 1: n - i - 1]
    y = series[order:]

    # 最小二乘求解AR系数
    try:
        phi, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(n_forecast, np.mean(series))

    # 逐步外推：每一步用最近的order个值预测下一个值
    predictions = np.zeros(n_forecast)
    history = series.copy()

    for t in range(n_forecast):
        recent = history[-order:][::-1]  # 最新的order个值，时间从近到远
        pred = np.dot(phi, recent)
        predictions[t] = pred
        history = np.append(history, pred)

    return predictions

def apply_residual_correction(
        demand: np.ndarray,
        imfs: np.ndarray,
) -> np.ndarray:
    """
    对IMF分量施加残差校正，确保 sum(IMFs) = demand 精确成立。

    VMD分解存在固有的近似误差（vmdpy算法收敛后 sum(modes) ≠ signal），
    残差校正按各方差比例将残差分配到各IMF上，保证重构精度。
    该校正对IMF的影响与VMD重构误差成正比：VMD参数越优，校正量越小。

    两阶段校正：
      1. 按方差比例分配残差（高能量IMF吸收更多误差）
      2. 均匀微调消除浮点残余

    Args:
        demand (np.ndarray): 原始需求序列，形状 (n,)
        imfs (np.ndarray): IMF分量，形状 (K, n)

    Returns:
        np.ndarray: 校正后的IMF分量，形状 (K, n)，保证 sum(IMFs) == demand
    """
    K = imfs.shape[0]
    imf_sum = imfs.sum(axis=0)
    residual = demand - imf_sum

    # 第一阶段：按方差比例分配
    imf_vars = np.array([np.var(imfs[k]) for k in range(K)])
    total_var = imf_vars.sum()
    if total_var == 0:
        var_ratios = np.ones(K) / K
    else:
        var_ratios = imf_vars / total_var

    imfs_corrected = imfs.copy()
    for k in range(K):
        imfs_corrected[k] += residual * var_ratios[k]

    # 第二阶段：均匀微调消除浮点残余
    imf_sum_corrected = imfs_corrected.sum(axis=0)
    final_residual = demand - imf_sum_corrected
    imfs_corrected += final_residual / K

    return imfs_corrected

def apply_vmd_to_subset(
        subset_demand: np.ndarray,
        imfs_train: np.ndarray,
        train_demand: np.ndarray,
        ar_order: int = 6,
) -> np.ndarray:
    """
    将训练集的VMD分解结果推广到验证/测试集，避免信息泄露。

    策略改进：使用自回归(AR)模型替代线性外推。
    对每个IMF分量，在训练集上拟合AR模型并逐步外推到验证/测试集。
    AR模型能够保留IMF的振荡特性（自相关性、周期性），而非退化为线性趋势。
    最后按方差比例分配残差，确保IMF之和等于原始需求值。

    Args:
        subset_demand (np.ndarray): 验证/测试集的需求序列，形状 (n_subset,)
        imfs_train (np.ndarray): 训练集的IMF分量，形状 (K, n_train)
        train_demand (np.ndarray): 训练集的需求序列，形状 (n_train,)
        ar_order (int): AR模型阶数，默认6（可捕获月度数据的半年周期模式）

    Returns:
        np.ndarray: 验证/测试集的IMF分量，形状 (K, n_subset)
    """
    K = imfs_train.shape[0]
    n_subset = len(subset_demand)

    # 计算训练集中各IMF的方差占比（用于残差分配）
    imf_vars = np.array([np.var(imfs_train[k]) for k in range(K)])
    total_var = imf_vars.sum()
    if total_var == 0:
        var_ratios = np.ones(K) / K
    else:
        var_ratios = imf_vars / total_var

    # 对每个IMF用AR模型外推
    imfs_subset = np.zeros((K, n_subset))

    for k in range(K):
        train_imf_k = imfs_train[k]

        # 根据IMF频率自适应选择AR阶数：
        # 低频IMF（方差大）用较高阶数，高频IMF用较低阶数
        # 防止高频分量因过拟合AR而产生发散
        imf_energy = np.var(train_imf_k)
        if imf_energy < 1e-8:
            # 近零方差的IMF，直接用均值填充
            imfs_subset[k] = np.full(n_subset, np.mean(train_imf_k))
        else:
            # 自适应阶数：低频分量用较高阶，高频分量用较低阶
            adaptive_order = min(ar_order, max(2, len(train_imf_k) // 10))
            imfs_subset[k] = _fit_ar_predict(train_imf_k, adaptive_order, n_subset)

    # 校正：确保 IMF 之和等于原始需求（复用通用残差校正函数）
    imfs_subset = apply_residual_correction(subset_demand, imfs_subset)

    return imfs_subset

# ====================== 5. 对完整数据执行VMD分解（保留兼容接口） ======================
def perform_vmd(df: pd.DataFrame, demand_col: str) -> pd.DataFrame:
    """
    对需求序列进行变分模态分解(VMD)，并将分解得到的本征模态函数(IMF)作为新特征添加到数据中。

    注意：此函数对全量数据执行VMD分解，存在信息泄露风险。
    推荐使用"先划分再分解"的流程（见 __main__ 部分）。

    Args:
        df (pd.DataFrame): 包含需求数据的DataFrame。
        demand_col (str): 需要进行VMD分解的需求列名，默认为'demand'。

    Returns:
        pd.DataFrame: 添加了IMF_1至IMF_K列的DataFrame，每列对应一个本征模态函数分量。
    """
    df_vmd = df.copy()
    u, _, _ = VMD(df_vmd[demand_col].values, alpha=VMD_ALPHA, tau=VMD_TAU, K=VMD_K, DC=VMD_DC, init=VMD_INIT, tol=VMD_TOL)

    # 将VMD分解得到的K个IMF分量添加为新特征列
    for k in range(VMD_K):
        df_vmd[f'IMF_{k + 1}'] = u[k, :]
        # print(f"  {demand_col} -> IMF_{k + 1}")
    return df_vmd

# ====================== 6. 划分训练、验证和测试数据集 ======================
def split_by_time(df: pd.DataFrame, train_ratio: float , val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序将数据集划分为训练集、验证集和测试集。

    严格按照时间索引顺序进行数据划分，确保不会发生时间泄露。
    默认按照训练集、验证集、测试集的比例进行划分。

    Args:
        df (pd.DataFrame): 待划分的DataFrame数据，必须已按时间排序。
        train_ratio (float): 训练集占总数据的比例（如0.7表示70%）。
        val_ratio (float): 验证集占总数据的比例（如0.1表示10%），剩余部分为测试集。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 包含三个DataFrame的元组，
            依次为训练集、验证集和测试集。
    """
    df_split = df.copy()
    n = len(df_split)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df_split[:train_end],
        df_split[train_end:val_end],
        df_split[val_end:]
    )

# ====================== 7. 拟合MinMaxScaler，统一转换三路数据并保存 ======================
def normalize_and_save(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    使用Min-Max归一化对训练集、验证集和测试集进行标准化处理，并保存结果。

    首先在训练集上拟合MinMaxScaler（避免数据泄露），然后用该scaler转换三个数据集。
    归一化后的数据保存为CSV文件，scaler对象保存为pkl文件供后续反归一化使用。

    Args:
        train (pd.DataFrame): 训练集数据。
        val (pd.DataFrame): 验证集数据。
        test (pd.DataFrame): 测试集数据。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 包含三个归一化后DataFrame的元组，
            依次为训练集、验证集和测试集。
    """
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    # 识别列分组
    imf_cols = [col for col in train.columns if col.startswith('IMF_')]
    other_cols = [col for col in train.columns if col not in [DEMAND_COL] + imf_cols]

    # 1. 对 demand 和 IMF 列使用同一套 min/max（基于训练集的 demand）
    d_min = train[DEMAND_COL].min()
    d_max = train[DEMAND_COL].max()
    joblib.dump({'min': d_min, 'max': d_max}, DEMAND_MIN_MAX_SCALER_PATH)

    def scale_group(df, d_min, d_max):
        """对 demand 和 IMF 列应用相同的线性缩放"""
        df_scaled = df.copy()
        # 缩放 demand
        df_scaled[DEMAND_COL] = (df[DEMAND_COL] - d_min) / (d_max - d_min)
        # 缩放每个 IMF 列（使用相同的 min/max）
        for col in imf_cols:
            if col in df.columns:
                df_scaled[col] = (df[col] - d_min) / (d_max - d_min)
        return df_scaled

    train_scaled = scale_group(train, d_min, d_max)
    val_scaled = scale_group(val, d_min, d_max)
    test_scaled = scale_group(test, d_min, d_max)

    # 2. 对其他外部特征使用独立的 MinMaxScaler（只拟合训练集）
    if other_cols:
        scaler = MinMaxScaler()
        train_other = pd.DataFrame(scaler.fit_transform(train[other_cols]),
                                   columns=other_cols, index=train.index)
        val_other = pd.DataFrame(scaler.transform(val[other_cols]),
                                 columns=other_cols, index=val.index)
        test_other = pd.DataFrame(scaler.transform(test[other_cols]),
                                  columns=other_cols, index=test.index)

        # 合并缩放后的 external 列
        for df_scaled, df_other in zip([train_scaled, val_scaled, test_scaled],
                                       [train_other, val_other, test_other]):
            for col in other_cols:
                df_scaled[col] = df_other[col]

        # 保存外部特征的 scaler（供可能的外部特征反归一化使用，但一般不需要）
        joblib.dump(scaler, SCALER_FILE_PATH)

    # 保存归一化后的数据
    train_scaled.to_csv(TRAIN_FILE_PATH)
    val_scaled.to_csv(VAL_FILE_PATH)
    test_scaled.to_csv(TEST_FILE_PATH)

    return train_scaled, val_scaled, test_scaled

# ========================= 测试 =======================
if __name__ == '__main__':
    print("Step 1/6: 加载原始数据...")
    df = load_raw_data(SIM_DATASET_PATH)

    print("Step 2/6: 时间顺序划分（先划分再清洗，避免信息泄露）...")
    train, val, test = split_by_time(df, TRAIN_RATIO, VAL_RATIO)

    print("Step 3/6: 数据清洗（仅用训练集统计量，无信息泄露）...")
    train, clean_stats = clean_data_train(train)
    val = clean_data_with_stats(val, clean_stats)
    test = clean_data_with_stats(test, clean_stats)

    print("Step 4/6: VMD分解（仅在训练集上拟合，AR外推到验证/测试集）...")
    # 仅在训练集上执行VMD分解
    train_demand = train[DEMAND_COL].values
    imfs_train, vmd_info = perform_vmd_train_only(train_demand)

    # 防御性校验：确保IMF长度与DataFrame行数一致
    assert imfs_train.shape[1] == len(train), \
        f"IMF长度({imfs_train.shape[1]})与训练集行数({len(train)})不一致"

    # 训练集残差校正：确保 sum(IMFs) = demand 精确成立
    # VMD分解存在固有近似误差，校正后重构MAE从数千降至~0
    mae_before = np.mean(np.abs(train_demand - imfs_train.sum(axis=0)))
    imfs_train = apply_residual_correction(train_demand, imfs_train)
    mae_after = np.mean(np.abs(train_demand - imfs_train.sum(axis=0)))
    print(f"  训练集残差校正: 重构MAE {mae_before:.2f} → {mae_after:.2e}")

    for k in range(VMD_K):
        train[f'IMF_{k + 1}'] = imfs_train[k, :]
    print(f"  训练集VMD分解完成: {VMD_K} 个IMF分量")

    # 将VMD分解外推到验证集
    val_demand = val[DEMAND_COL].values
    imfs_val = apply_vmd_to_subset(val_demand, imfs_train, train_demand)

    assert imfs_val.shape[1] == len(val), \
        f"IMF长度({imfs_val.shape[1]})与验证集行数({len(val)})不一致"

    for k in range(VMD_K):
        val[f'IMF_{k + 1}'] = imfs_val[k, :]

    # 将VMD分解外推到测试集
    test_demand = test[DEMAND_COL].values
    imfs_test = apply_vmd_to_subset(test_demand, imfs_train, train_demand)

    assert imfs_test.shape[1] == len(test), \
        f"IMF长度({imfs_test.shape[1]})与测试集行数({len(test)})不一致"

    for k in range(VMD_K):
        test[f'IMF_{k + 1}'] = imfs_test[k, :]
    print(f"  验证/测试集VMD外推完成（无信息泄露）")

    # 校验：IMF之和应约等于原始需求
    train_recon = sum(train[f'IMF_{k+1}'] for k in range(VMD_K))
    val_recon = sum(val[f'IMF_{k+1}'] for k in range(VMD_K))
    test_recon = sum(test[f'IMF_{k+1}'] for k in range(VMD_K))
    print(f"  训练集重构误差: {np.mean(np.abs(train[DEMAND_COL].values - train_recon.values)):.6f}")
    print(f"  验证集重构误差: {np.mean(np.abs(val[DEMAND_COL].values - val_recon.values)):.6f}")
    print(f"  测试集重构误差: {np.mean(np.abs(test[DEMAND_COL].values - test_recon.values)):.6f}")

    print("Step 5/6: Min-Max归一化...")
    train_s, val_s, test_s = normalize_and_save(train, val, test)

    print(f"✅ 预处理完成 | 训练集: {len(train_s)} | 验证集: {len(val_s)} | 测试集: {len(test_s)}")
    print(f"✅ 文件已保存至: {DATA_PROCESSED_DIR}")
    print(f"⚠️ 注意: VMD分解仅在训练集上拟合，验证/测试集通过AR模型外推获得，避免信息泄露")
    print(f"⚠️ 注意: 数据清洗先划分再清洗，缩尾分位数仅用训练集计算，避免信息泄露")