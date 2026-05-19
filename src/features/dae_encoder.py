"""
文件位置: src/features/dae_encoder.py
作用: 去噪自编码器(DAE)——对高维特征进行降维与去噪
所包含的函数/类:
  - DenoisingAE(nn.Module): DAE网络结构
  - train_dae(): 训练流程（加噪 → 重构 → MSELoss）
  - encode_and_save(): 用训练好的编码器对train/val/test编码并保存
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 读取 data/processed/{train,val,test}.csv, data/features/selected_features.json
  - 输出 data/processed/{train,val,test}_encoded.csv, checkpoints/dae_model.pth
"""

# ======================== 0. 引入包 =======================
import torch.nn as nn
import os
import torch
import json
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from configs.config import (
    NOISE_STD,                          # 噪声标准差
    FEATURE_SELECTION_FILE_PATH,        # 特征筛选结果文件
    TRAIN_FILE_PATH,                    # 训练集文件路径
    VAL_FILE_PATH,                      # 验证集文件路径
    TEST_FILE_PATH,                     # 测试集文件路径
    DAE_MODEL_FILE_PATH,                # DAE 模型保存路径
    MODELS_DIR,                         # 模型保存目录
    DATA_PROCESSED_DIR,                 # 数据处理目录
    DATA_FIGURES_DIR,                   # 绘图数据存放目录
    DAE_EPOCHS,                         # DAE 训练轮数
    DAE_BATCH_SIZE,                     # 训练批次大小
    DAE_LEARNING_RATE,                  # 学习率
    DAE_HIDDEN_DIM,                     # 隐藏层维度
    TOP_N,                              # 随机森林特征重要性评估的top N
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================== 1. DAE网络结构 =======================
class DenoisingAE(nn.Module):
    """
    去噪自编码器（Denoising Autoencoder）模型。

    该模型通过在输入数据中添加高斯噪声来训练自编码器，使其能够学习数据的鲁棒特征表示。
    包含编码器和解码器两个部分，编码器将输入压缩为低维表示，解码器尝试从编码中重构原始输入。
    """
    def __init__(self, input_dim: int, hidden_dim: list, noise_std: float, dropout: float = 0.2):
        """
        初始化去噪自编码器模型。

        Args:
            input_dim (int): 输入特征的维度。
            hidden_dim (list): 隐藏层维度列表，定义编码器的层级结构。
            noise_std (float): 添加到输入数据的高斯噪声标准差。
        """
        super(DenoisingAE, self).__init__()
        self.noise_std = noise_std
        dims = [input_dim] + hidden_dim
        dims_rev = dims[::-1]
        # 编码器
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.encoder = nn.Sequential(*layers)
        # 解码器
        layers = []
        for i in range(len(dims_rev) - 1):
            layers.extend([
                nn.Linear(dims_rev[i], dims_rev[i + 1]),
                nn.BatchNorm1d(dims_rev[i + 1]),
                nn.ReLU(),
            ])
            if i < len(dims_rev) - 2:
                layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*layers)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        向输入数据添加高斯噪声。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 添加噪声后的张量。
        """
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def forward(self, x: torch.Tensor, noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播过程。

        在训练模式下可选择性地添加噪声，在推理模式下直接使用输入数据。
        返回编码后的特征表示和重构的输出。

        Args:
            x (torch.Tensor): 输入张量。
            noise (bool): 是否添加噪声，默认为True。仅在训练模式下生效。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 包含两个元素的元组：
                - x_encoded: 编码后的特征表示
                - x_decoded: 解码器重构的输出
        """
        if self.training and noise:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x

        x_encoded = self.encoder(x_noisy)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded

# ======================== 2. 训练流程（加噪 → 重构 → MSELoss） =======================
def train_dae(epochs: int, batch_size: int, lr: float, noise_std: float, hidden_dim: list) -> None:
    """
    训练去噪自编码器（DAE）模型并对数据集进行编码。

    该函数执行完整的DAE训练流程：加载已筛选的特征数据，构建训练和验证集，
    训练模型以学习数据的鲁棒特征表示，保存最佳模型状态和训练损失曲线，
    最后使用训练好的模型对训练集、验证集和测试集进行编码并保存结果。

    Args:
        epochs (int): 训练轮数。
        batch_size (int): 批量大小。
        lr (float): 学习率。
        noise_std (float): 添加到输入数据的高斯噪声标准差。

    Returns:
        None: 该函数不返回值，但会保存模型、损失数据和编码后的数据集到文件系统。
    """
    # 加载筛选后的特征
    with open(FEATURE_SELECTION_FILE_PATH, 'r') as f:
        selected_features = json.load(f)
    print(f"🚀 读取筛选后的特征: {selected_features}")
    # print(type(selected_features))
    df_train = pd.read_csv(TRAIN_FILE_PATH, index_col='date', parse_dates=True)
    df_val = pd.read_csv(VAL_FILE_PATH, index_col='date', parse_dates=True)
    x_train = df_train[selected_features].values
    x_val = df_val[selected_features].values

    # 转换为 tensor
    x_train_tensor = torch.FloatTensor(x_train).to(device)
    x_val_tensor = torch.FloatTensor(x_val).to(device)
    print(f"🚀 训练集和验证集的形状: {x_train_tensor.shape}, {x_val_tensor.shape}")

    train_loader = DataLoader(TensorDataset(x_train_tensor), batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None

    input_dim = x_train.shape[1]

    model = DenoisingAE(input_dim=input_dim, hidden_dim=hidden_dim, noise_std=noise_std).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("🚀 开始训练 DAE...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            x_encoded, x_decoded = model(x, True)
            loss = criterion(x_decoded, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            x_encoded, x_decoded = model(x_val_tensor, False)  # 验证时不加噪声
            val_loss = criterion(x_decoded, x_val_tensor).item()
            val_losses.append(val_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)

    # 保存损失函数
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
    losses_df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
    losses_df.to_csv(os.path.join(DATA_FIGURES_DIR, 'losses.csv'))

    encode_and_save(model, TRAIN_FILE_PATH, selected_features, 'train')
    encode_and_save(model, VAL_FILE_PATH, selected_features, 'val')
    encode_and_save(model, TEST_FILE_PATH, selected_features, 'test')

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.cpu()
    torch.save(model.state_dict(), DAE_MODEL_FILE_PATH)
    print(f"✅ DAE模型已保存: {DAE_MODEL_FILE_PATH}")

# ========================== 3. 编码和保存 =======================
def encode_and_save(model: nn.Module, file_path: str, selected_features: list, split_name: str) -> None:
    """
    使用训练好的模型对数据进行编码并保存结果。

    加载原始数据，提取选定特征，通过模型编码器生成低维特征表示，
    然后将编码后的特征与未选中的原始特征合并保存为新的CSV文件。

    Args:
        model (nn.Module): 已训练好的去噪自编码器模型。
        file_path (str): 原始数据CSV文件的路径。
        selected_features (list): 用于编码的特征列名列表。
        split_name (str): 数据集分割名称（如'train'、'val'、'test'），用于命名输出文件。

    Returns:
        None: 该函数不返回值，但会将编码后的数据保存到文件系统中。
    """
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    x = df[selected_features].values

    x_tensor = torch.FloatTensor(x).to(device)
    model.eval()
    with torch.no_grad():
        x_encoded, x_decoded = model(x_tensor, False)

    enc_df = pd.DataFrame(
        x_encoded.cpu().numpy(),
        columns=[f'ENC_{i}' for i in range(x_encoded.shape[1])],
        index=df.index
    )

    other_cols = [col for col in df.columns if col not in selected_features]
    final_df = pd.concat([enc_df, df[other_cols]], axis=1)

    save_path = os.path.join(DATA_PROCESSED_DIR, f'{split_name}_encoded.csv')
    final_df.to_csv(save_path)
    print(f"  {split_name}_encoded.csv | shape: {final_df.shape}")

# ========================= 测试 =======================
if __name__ == '__main__':
    model = DenoisingAE(input_dim=TOP_N, hidden_dim=DAE_HIDDEN_DIM, noise_std=NOISE_STD)
    train_dae(epochs=DAE_EPOCHS, batch_size=DAE_BATCH_SIZE, lr=DAE_LEARNING_RATE, noise_std=NOISE_STD, hidden_dim=DAE_HIDDEN_DIM)