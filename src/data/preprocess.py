"""
文件位置: src/data/preprocess.py
作用: 数据清洗、VMD分解、缺失值插补、异常值缩尾、Min-Max归一化
所包含的函数:
  - load_raw_data(): 加载原始仿真数据
  - clean_data(): 缺失值线性插值 + demand列1%-99%缩尾
  - perform_vmd(): 对demand序列做VMD分解，追加IMF列
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
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

# ====================== 2. 缺失值线性插值 + demand列1%-99%缩尾 ======================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    对数据进行清洗，包括缺失值插补和异常值缩尾处理。

    首先使用线性插值填补缺失值，并对首尾可能存在的缺失值进行前向和后向填充。
    然后对所有数值列进行1%-99%分位数的缩尾处理，以去除极端异常值。

    Args:
        df (pd.DataFrame): 待清洗的DataFrame数据。

    Returns:
        pd.DataFrame: 经过缺失值插补和异常值缩尾处理后的DataFrame。
    """
    df_clean = df.copy()
    df_clean = df_clean.interpolate(method='linear')
    df_clean = df_clean.ffill().bfill()

    # 对所有数值列进行1%-99%分位数缩尾处理，去除极端异常值
    num_cols = df_clean.select_dtypes(include=['number']).columns
    for col in num_cols:
        lower, upper = df_clean[col].quantile([0.01, 0.99])
        # print(f"  {col}: {lower:.2f} ~ {upper:.2f}")
        df_clean[col] = df_clean[col].clip(lower, upper)
    return df_clean

# ====================== 3. 对demand序列做VMD分解，追加IMF列 ======================
def perform_vmd(df: pd.DataFrame, demand_col: str) -> pd.DataFrame:
    """
    对需求序列进行变分模态分解(VMD)，并将分解得到的本征模态函数(IMF)作为新特征添加到数据中。

    使用VMD算法将原始需求信号分解为K个本征模态函数分量，每个分量代表不同频率尺度的波动特征，
    有助于捕捉时间序列中的多尺度模式和非线性特征。

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

# ====================== 4. 划分训练、验证和测试数据集 ======================
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

# ====================== 5. 拟合MinMaxScaler，统一转换三路数据并保存 ======================
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
    scaler = MinMaxScaler()
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    train_scaler = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    val_scaler = pd.DataFrame(scaler.transform(val), columns=val.columns, index=val.index)
    test_scaler = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

    train_scaler.to_csv(TRAIN_FILE_PATH)
    val_scaler.to_csv(VAL_FILE_PATH)
    test_scaler.to_csv(TEST_FILE_PATH)
    joblib.dump(scaler, SCALER_FILE_PATH)

    demand_min = train[DEMAND_COL].min()
    demand_max = train[DEMAND_COL].max()

    joblib.dump({'min': demand_min, 'max': demand_max}, DEMAND_MIN_MAX_SCALER_PATH)

    return (
        train_scaler,
        val_scaler,
        test_scaler
    )

# ========================= 测试 =======================
if __name__ == '__main__':
    print("Step 1/4: 加载原始数据...")
    df = load_raw_data(SIM_DATASET_PATH)

    print("Step 2/4: 数据清洗...")
    df = clean_data(df)

    print("Step 3/4: VMD分解并追加IMF特征...")
    df = perform_vmd(df, DEMAND_COL)
    print(f"  当前特征维度: {df.shape[1]} (含 {VMD_K} 个IMF分量)")

    print("Step 4/4: 时间顺序划分 + Min-Max归一化...")
    train, val, test = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_s, val_s, test_s = normalize_and_save(train, val, test)

    print(f"✅ 预处理完成 | 训练集: {len(train_s)} | 验证集: {len(val_s)} | 测试集: {len(test_s)}")
    print(f"✅ 文件已保存至: {DATA_PROCESSED_DIR}")