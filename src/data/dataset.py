"""
文件位置: src/data/dataset.py
作用: 电网物资需求预测数据集构建与加载
      将预处理后的CSV文件加载为PyTorch Dataset，自动识别IMF分量列（IMF_*）和DAE编码特征列（ENC_*），
      按滑动窗口生成(历史序列, 外部特征) → 未来需求 的样本，最后封装为DataLoader供模型训练/验证/测试。
所包含的类/函数:
  - GridMaterialDataset(Dataset): 自定义数据集类，按时间顺序生成固定长度的输入-目标样本对
  - get_dataloaders(): 工厂函数，一次性创建训练/验证/测试的DataLoader并返回(loader, n_imf, n_ext)
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 读取 data/processed/{train,val,test}_encoded.csv (需包含 IMF_*, ENC_*, demand 列)
  - 输出 DataLoader 对象，供 src/models/trainer.py 直接使用
  - 通过返回值 n_imf, n_ext 传递模型结构参数给 VMDCATBiLSTM
"""

# ======================== 0. 引入包 =======================
import os
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from configs.config import (
    DATA_PROCESSED_DIR,                 # 数据处理结果目录
    BATCH_SIZE,                         # 批次大小
    DEMAND_COL,                         # 目标列名
    SEQ_LEN,                            # 滑动窗口长度
    FEATURE_SELECTION_FILE_PATH,        # 特征筛选结果文件路径
)

# ======================== 1. 获取数据集 =======================
class GridMaterialDataset(Dataset):
    """
    电网物资需求预测数据集类

    基于PyTorch的Dataset类，支持三种数据模式，用于加载和处理VMD分解后的IMF数据和外部特征。
    支持滑动窗口采样，将时间序列数据转换为适合模型输入的样本格式。

    数据模式:
        - 'vmd_dae' (默认): 从 *_encoded.csv 读取 IMF_* + ENC_* 列，返回 (imfs, externals, y)
        - 'no_vmd': 从 *_encoded.csv 读取 demand + ENC_* 列合并为 x_combined，返回 (x_combined, y)
        - 'no_vmd_raw': 从原始CSV读取 demand + 原始选中特征列合并为 x_combined，返回 (x_combined, y)
        - 'no_dae': 从原始CSV读取 IMF_* 列，并从 selected_features.json 获取原始特征列名，返回 (imfs, externals, y)

    Attributes:
        seq_len (int): 输入序列长度（滑动窗口大小）
        pred_len (int): 预测步长，通常为1
        target_col (str): 目标列名称
        data_mode (str): 数据模式，'vmd_dae' / 'no_vmd' / 'no_dae'
        df (pd.DataFrame): 原始数据框，以日期为索引
        imf_cols (list[str]): IMF列名列表，按字母排序
        ext_cols (list[str]): 外部特征列名列表，按字母排序
        imf_data (np.ndarray): IMF数据数组，形状为 (n_samples, n_imf)
        ext_data (np.ndarray): 外部特征数据数组，形状为 (n_samples, n_ext)
        target_data (np.ndarray): 目标值数组，形状为 (n_samples,)
        combined_data (np.ndarray): 合并后的数据数组（仅 no_vmd 模式），形状为 (n_samples, 1+n_ext)
        n_imf (int): IMF分量数量
        n_ext (int): 外部特征数量
    """

    def __init__(
            self,
            csv_path: str,
            target_col: str,
            seq_len: int = 12,
            pred_len: int = 1,
            data_mode: str = 'vmd_dae',
            ext_cols_override: list = None,
    ):
        """
        初始化数据集

        读取CSV文件，根据data_mode自动识别列并进行数据验证和预处理。

        Args:
            csv_path (str): CSV文件路径
            target_col (str): 目标变量列名
            seq_len (int, optional): 输入序列长度，默认为12
            pred_len (int, optional): 预测步长，默认为1
            data_mode (str, optional): 数据模式，可选 'vmd_dae' / 'no_vmd' / 'no_dae'，默认 'vmd_dae'
            ext_cols_override (list, optional): 自定义外部特征列名列表，用于 no_dae 模式指定原始特征列

        Raises:
            AssertionError: 当缺少必要列时抛出异常
            ValueError: 当data_mode不合法时抛出异常
        """

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.data_mode = data_mode

        self.df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')

        # 根据data_mode识别列
        if data_mode == 'vmd_dae':
            # 标准模式：从 *_encoded.csv 读取 IMF_* + ENC_* 列，返回 (imfs, externals, y)
            self.imf_cols = sorted([col for col in self.df.columns if col.startswith('IMF_')])
            self.ext_cols = sorted([col for col in self.df.columns if col.startswith('ENC_')])

            assert len(self.imf_cols) > 0, f"❌ 未在 {csv_path} 中找到 IMF 列！(data_mode='vmd_dae')"
            assert len(self.ext_cols) > 0, f"❌ 未在 {csv_path} 中找到 ENC 列！(data_mode='vmd_dae')"

            self.imf_data = self.df[self.imf_cols].values
            self.ext_data = self.df[self.ext_cols].values
            self.n_imf = len(self.imf_cols)
            self.n_ext = len(self.ext_cols)

        elif data_mode == 'no_vmd':
            # 仅IMF模式：从 *_encoded.csv 读取 demand + ENC_* 列合并为 x_combined，返回 (x_combined, y)
            self.ext_cols = sorted([col for col in self.df.columns if col.startswith('ENC_')])

            assert len(self.ext_cols) > 0, f"❌ 未在 {csv_path} 中找到 ENC 列！(data_mode='no_vmd')"
            assert target_col in self.df.columns, f"❌ 未找到目标列 {target_col}！"

            demand_data = self.df[[target_col]].values.reshape(-1, 1)
            ext_data = self.df[self.ext_cols].values
            if ext_data.ndim == 1:
                ext_data = ext_data.reshape(-1, 1)

            self.combined_data = np.hstack([demand_data, ext_data])
            self.n_imf = 0
            self.n_ext = len(self.ext_cols)

        elif data_mode == 'no_vmd_raw':
            # 无VMD + 原始特征模式：demand + 原始选中特征合并为单一输入
            # （用于端到端DAE训练，模型内部完成特征编码）
            if ext_cols_override is not None:
                self.ext_cols = ext_cols_override
            else:
                sf_path = FEATURE_SELECTION_FILE_PATH
                if os.path.exists(sf_path):
                    with open(sf_path, 'r') as f:
                        self.ext_cols = json.load(f)
                else:
                    raise FileNotFoundError(
                        f"❌ no_vmd_raw 模式需要指定 ext_cols_override 或存在 {sf_path}"
                    )

            missing_cols = [col for col in self.ext_cols if col not in self.df.columns]
            assert len(missing_cols) == 0, \
                f"❌ 以下特征列在 {csv_path} 中不存在: {missing_cols}"
            assert target_col in self.df.columns, f"❌ 未找到目标列 {target_col}！"

            demand_data = self.df[[target_col]].values.reshape(-1, 1)
            ext_data = self.df[self.ext_cols].values
            self.combined_data = np.hstack([demand_data, ext_data])
            self.n_imf = 0
            self.n_ext = len(self.ext_cols)

        elif data_mode == 'no_dae':
            # 仅外部特征模式：从原始 *_encoded.csv 或 *_encoded.csv 读取 IMF_* 列，从 selected_features.json 获取原始特征列名，返回 (imfs, externals, y)
            self.imf_cols = sorted([col for col in self.df.columns if col.startswith('IMF_')])

            assert len(self.imf_cols) > 0, f"❌ 未在 {csv_path} 中找到 IMF 列！(data_mode='no_dae')"

            if ext_cols_override is not None:
                self.ext_cols = ext_cols_override
            else:
                sf_path = FEATURE_SELECTION_FILE_PATH
                if os.path.exists(sf_path):
                    with open(sf_path, 'r') as f:
                        self.ext_cols = json.load(f)
                else:
                    raise FileNotFoundError(
                        f"❌ no_dae 模式需要指定 ext_cols_override 或存在 {sf_path}"
                    )

            missing_cols = [col for col in self.ext_cols if col not in self.df.columns]
            assert len(missing_cols) == 0, \
                f"❌ 以下外部特征列在 {csv_path} 中不存在: {missing_cols}。" \
                f"\n   提示: no_dae 模式需读取未编码的CSV文件（如 train.csv 而非 train_encoded.csv）"

            self.imf_data = self.df[self.imf_cols].values
            self.ext_data = self.df[self.ext_cols].values
            self.n_imf = len(self.imf_cols)
            self.n_ext = len(self.ext_cols)
        else:
            raise ValueError(f"❌ 不支持的数据模式: {data_mode}，请选择 'vmd_dae' / 'no_vmd' / 'no_vmd_raw' / 'no_dae'")

            assert target_col in self.df.columns, f"❌ 未找到目标列 {target_col}！"

        self.target_data = self.df[target_col].values

    def __len__(self):
        """
        返回数据集的样本总数

        根据序列长度和预测步长计算有效样本数，确保最后一个样本有完整的输入输出。

        Returns:
            int: 可用样本数量
        """
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        """
        获取指定索引的单个样本

        根据数据模式返回不同格式的样本：
        - vmd_dae / no_dae: 返回 (imfs_tensor, exts_tensor, y_tensor)
        - no_vmd: 返回 (x_combined_tensor, y_tensor)

        Args:
            idx (int): 样本索引

        Returns:
            tuple: 根据数据模式返回不同格式的样本
        """
        start_idx = idx
        end_idx = idx + self.seq_len
        target_idx = end_idx + self.pred_len - 1

        y = self.target_data[target_idx]
        y_tensor = torch.FloatTensor([y])

        if self.data_mode in ('no_vmd', 'no_vmd_raw'):
            x_combined = self.combined_data[start_idx:end_idx]
            x_combined_tensor = torch.FloatTensor(x_combined)
            return x_combined_tensor, y_tensor
        else:
            # vmd_dae 或 no_dae 模式
            imfs = self.imf_data[start_idx:end_idx]
            exts = self.ext_data[start_idx:end_idx]
            imfs_tensor = torch.FloatTensor(imfs)
            exts_tensor = torch.FloatTensor(exts)
            return imfs_tensor, exts_tensor, y_tensor

# ======================== 2. 数据集加载器 =======================
def get_dataloaders(
        data_dir: str,
        target_col: str,
        seq_len: int = 12,
        pred_len: int = 1,
        batch_size: int = 32,
        data_mode: str = 'vmd_dae',
        ext_cols_override: list = None,
        file_suffix: str = '_encoded'
):
    """
    创建训练、验证和测试数据加载器

    根据data_mode自动选择CSV文件后缀和解析逻辑，构建DataLoader实例。

    Args:
        data_dir (str): 数据目录路径
        target_col (str): 目标变量列名
        seq_len (int, optional): 输入序列长度，默认为12
        pred_len (int, optional): 预测步长，默认为1
        batch_size (int, optional): 批次大小，默认为32
        data_mode (str, optional): 数据模式，'vmd_dae' / 'no_vmd' / 'no_dae'
        ext_cols_override (list, optional): no_dae模式下的外部特征列名列表
        file_suffix (str, optional): CSV文件后缀，默认'_encoded'。no_dae模式下自动切换为''

    Returns:
        tuple:
            - train_loader (DataLoader): 训练数据加载器
            - val_loader (DataLoader): 验证数据加载器
            - test_loader (DataLoader): 测试数据加载器
            - n_imf (int): IMF分量数量（no_vmd模式下为0）
            - n_ext (int): 外部特征数量
    """
    # no_dae / no_vmd_raw 模式需要读取未编码的CSV文件（保留原始特征列）
    actual_suffix = '' if data_mode in ('no_dae', 'no_vmd_raw') else file_suffix

    train_path = os.path.join(data_dir, f'train{actual_suffix}.csv')
    val_path = os.path.join(data_dir, f'val{actual_suffix}.csv')
    test_path = os.path.join(data_dir, f'test{actual_suffix}.csv')

    # 检查文件是否存在
    for p in [train_path, val_path, test_path]:
        assert os.path.exists(p), f"❌ 数据文件不存在: {p}"

    common_kwargs = dict(
        target_col=target_col,
        seq_len=seq_len,
        pred_len=pred_len,
        data_mode=data_mode,
        ext_cols_override=ext_cols_override,
    )

    train_dataset = GridMaterialDataset(train_path, **common_kwargs)
    val_dataset = GridMaterialDataset(val_path, **common_kwargs)
    test_dataset = GridMaterialDataset(test_path, **common_kwargs)

    n_imf = train_dataset.n_imf
    n_ext = train_dataset.n_ext

    print(f"📊 数据集构建完成 | 模式: {data_mode} | IMF数: {n_imf}, 外部特征数: {n_ext}")
    print(f"   训练集样本数: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_imf, n_ext

# ========================= 测试代码 =======================
if __name__ == '__main__':
    import numpy as np

    print("=" * 60)
    print("🧪 测试 vmd_dae 模式 (默认)")
    print("=" * 60)
    train_loader, val_loader, test_loader, n_imf, n_ext = get_dataloaders(
        data_dir=DATA_PROCESSED_DIR,
        target_col=DEMAND_COL,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        data_mode='vmd_dae',
    )
    for batch in train_loader:
        imfs_batch, exts_batch, y_batch = batch
        print(f"   imfs: {imfs_batch.shape} | exts: {exts_batch.shape} | y: {y_batch.shape}")
        break

    print("\n" + "=" * 60)
    print("🧪 测试 no_vmd 模式")
    print("=" * 60)
    train_loader, val_loader, test_loader, n_imf, n_ext = get_dataloaders(
        data_dir=DATA_PROCESSED_DIR,
        target_col=DEMAND_COL,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        data_mode='no_vmd',
    )
    for batch in train_loader:
        x_batch, y_batch = batch
        print(f"   x_combined: {x_batch.shape} | y: {y_batch.shape}")
        break

    print("\n" + "=" * 60)
    print("🧪 测试 no_dae 模式")
    print("=" * 60)
    train_loader, val_loader, test_loader, n_imf, n_ext = get_dataloaders(
        data_dir=DATA_PROCESSED_DIR,
        target_col=DEMAND_COL,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        data_mode='no_dae',
    )
    for batch in train_loader:
        imfs_batch, exts_batch, y_batch = batch
        print(f"   imfs: {imfs_batch.shape} | exts: {exts_batch.shape} | y: {y_batch.shape}")
        break