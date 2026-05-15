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
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from configs.config import (
    DATA_PROCESSED_DIR,                 # 数据处理结果目录
    BATCH_SIZE,                         # 批次大小
    DEMAND_COL,                         # 目标列名
    SEQ_LEN,                            # 滑动窗口长度
)


# ======================== 1. 获取数据集 =======================
class GridMaterialDataset(Dataset):
    """
    电网物资需求预测数据集类

    基于PyTorch的Dataset类，用于加载和处理VMD分解后的IMF数据和编码后的外部特征。
    支持滑动窗口采样，将时间序列数据转换为适合模型输入的样本格式。
    每个样本包含：IMF分量序列、外部特征序列和对应的目标值。

    Attributes:
        seq_len (int): 输入序列长度（滑动窗口大小）
        pred_len (int): 预测步长，通常为1
        target_col (str): 目标列名称
        df (pd.DataFrame): 原始数据框，以日期为索引
        imf_cols (list[str]): IMF列名列表，按字母排序
        ext_cols (list[str]): 外部特征列名列表（ENC_前缀），按字母排序
        imf_data (np.ndarray): IMF数据数组，形状为 (n_samples, n_imf)
        ext_data (np.ndarray): 外部特征数据数组，形状为 (n_samples, n_ext)
        target_data (np.ndarray): 目标值数组，形状为 (n_samples,)
        n_imf (int): IMF分量数量
        n_ext (int): 外部特征数量
    """

    def __init__(self, csv_path: str, target_col: str, seq_len: int = 12, pred_len: int = 1):
        """
        初始化数据集

        读取CSV文件，自动识别IMF和外部特征列，进行数据验证和预处理。

        Args:
            csv_path (str): CSV文件路径，需包含date列、IMF_*列、ENC_*列和目标列
            target_col (str): 目标变量列名
            seq_len (int, optional): 输入序列长度，默认为12
            pred_len (int, optional): 预测步长，默认为1

        Raises:
            AssertionError: 当缺少IMF列、ENC列或目标列时抛出异常
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col

        self.df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')

        # 自动识别并排序IMF和外部特征列
        self.imf_cols = sorted([col for col in self.df.columns if col.startswith('IMF_')])
        self.ext_cols = sorted([col for col in self.df.columns if col.startswith('ENC_')])

        assert len(self.imf_cols) > 0, f"❌ 未在 {csv_path} 中找到 IMF 列！"
        assert len(self.ext_cols) > 0, f"❌ 未在 {csv_path} 中找到 ENC 列！"
        assert self.target_col in self.df.columns, f"❌ 未找到目标列 {self.target_col}！"

        self.imf_data = self.df[self.imf_cols].values
        self.ext_data = self.df[self.ext_cols].values
        self.target_data = self.df[self.target_col].values

        self.n_imf = len(self.imf_cols)
        self.n_ext = len(self.ext_cols)

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

        使用滑动窗口方式提取数据：从idx开始取seq_len长度的序列作为输入，
        在end_idx + pred_len - 1位置的值作为预测目标。

        Args:
            idx (int): 样本索引，从0开始

        Returns:
            tuple:
                - imfs_tensor (torch.Tensor): IMF分量序列，形状为 (seq_len, n_imf)
                - exts_tensor (torch.Tensor): 外部特征序列，形状为 (seq_len, n_ext)
                - y_tensor (torch.Tensor): 目标值，形状为 (1,)
        """
        start_idx = idx
        end_idx = idx + self.seq_len
        target_idx = end_idx + self.pred_len - 1

        imfs = self.imf_data[start_idx:end_idx]
        exts = self.ext_data[start_idx:end_idx]
        y = self.target_data[target_idx]

        imfs_tensor = torch.FloatTensor(imfs)
        exts_tensor = torch.FloatTensor(exts)
        y_tensor = torch.FloatTensor([y])

        return imfs_tensor, exts_tensor, y_tensor


# ======================== 2. 数据集加载器 =======================
def get_dataloaders(
        data_dir: str,
        target_col: str,
        seq_len: int = 12,
        pred_len: int = 1,
        batch_size: int = 32,
):
    """
    创建训练、验证和测试数据加载器

    从指定的数据目录加载三个数据集分割（train/val/test），构建DataLoader实例。
    自动提取IMF数量和外部特征数量，供模型初始化使用。
    训练集启用随机打乱和丢弃最后一个不完整batch，验证集和测试集保持顺序。

    Args:
        data_dir (str): 数据目录路径，应包含train_encoded.csv、val_encoded.csv、test_encoded.csv
        target_col (str): 目标变量列名
        seq_len (int, optional): 输入序列长度，默认为12
        pred_len (int, optional): 预测步长，默认为1
        batch_size (int, optional): 批次大小，默认为32

    Returns:
        tuple:
            - train_loader (DataLoader): 训练数据加载器，shuffle=True, drop_last=True
            - val_loader (DataLoader): 验证数据加载器，shuffle=False
            - test_loader (DataLoader): 测试数据加载器，shuffle=False
            - n_imf (int): IMF分量数量
            - n_ext (int): 外部特征数量
    """
    train_path = os.path.join(data_dir, 'train_encoded.csv')
    val_path = os.path.join(data_dir, 'val_encoded.csv')
    test_path = os.path.join(data_dir, 'test_encoded.csv')

    train_dataset = GridMaterialDataset(train_path, target_col, seq_len=seq_len, pred_len=pred_len)
    val_dataset = GridMaterialDataset(val_path, target_col, seq_len=seq_len, pred_len=pred_len)
    test_dataset = GridMaterialDataset(test_path, target_col, seq_len=seq_len, pred_len=pred_len)

    n_imf = train_dataset.n_imf
    n_ext = train_dataset.n_ext

    print(f"📊 数据集构建完成 | IMF数: {n_imf}, 外部特征数: {n_ext}")
    print(f"   训练集样本数: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_imf, n_ext


# ========================= 测试代码 =======================
if __name__ == '__main__':
    print("🚀 开始测试数据加载器...")

    train_loader, val_loader, test_loader, n_imf, n_ext = get_dataloaders(
        data_dir=DATA_PROCESSED_DIR,
        target_col=DEMAND_COL,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE
    )

    # 取出一个 batch 查看形状
    for imfs_batch, exts_batch, y_batch in train_loader:
        print("\n✅ 成功获取一个 Batch:")
        print(f"   imfs_batch shape: {imfs_batch.shape}   (Batch, Seq_Len, N_IMF)")
        print(f"   exts_batch shape: {exts_batch.shape}  (Batch, Seq_Len, N_External)")
        print(f"   y_batch shape:    {y_batch.shape}           (Batch, 1)")

        # 验证能否直接输入模型
        # model = VMDCATBiLSTM(n_imf=n_imf, n_external=n_ext)
        # pred, weights = model(imfs_batch, exts_batch)
        # print(f"   Model output pred shape: {pred.shape}")
        break