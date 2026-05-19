
# ======================== 0. 引入包 =======================
import os
import json
import time
import torch
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from src.models.trainer import calculate_metrics
from configs.config import (
    DATA_PROCESSED_DIR,                 # 处理后数据目录
    FEATURE_SELECTION_FILE_PATH,        # 特征筛选结果文件路径
    DEMAND_COL,                         # 目标列名
    BATCH_SIZE,                         # 批次大小
    MODELS_DIR,                         # 模型保存目录
    DEMAND_MIN_MAX_SCALER_PATH,         # 需求归一化参数文件路径
    SEQ_LEN,                            # 滑动窗口长度
    DATA_FIGURES_DIR,                   # 绘图数据保存目录
)

# ======================== 1. 配置 =======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_LAYERS = 2
DROPOUT = 0.4
LEARNING_RATE = 1e-3
EPOCHS = 300
PATIENCE = 300
USE_EXTERNAL_FEATURES = False
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
models_file_path = os.path.join(MODELS_DIR, f"best_lstm_model.pth")
result_file_path = os.path.join(DATA_FIGURES_DIR, 'lstm_results.csv')

# ======================== 2. 数据处理 =======================
def create_sequences(data, seq_len):
    """
    将时间序列数据转换为监督学习格式的序列样本。

    参数:
        data: 输入的时间序列数据数组，形状为 (样本数, 特征数)
        seq_len: 序列窗口长度，即每个样本包含的历史时间步数

    返回:
        x: 特征序列数组，形状为 (样本数-seq_len, seq_len, 特征数)
        y: 目标值数组，形状为 (样本数-seq_len, 1)，对应每个序列的下一个时间步的目标值
    """
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def load_lstm_data(seq_len: int = 12, use_external: bool = True):
    """
    加载并预处理LSTM模型所需的数据，包括读取数据、构建序列和创建数据加载器。

    参数:
        seq_len: 序列窗口长度，用于创建滑动窗口样本
        use_external: 是否使用外部特征，如为True则加载选定的外部特征

    返回:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        test_loader: 测试集数据加载器
        input_dim: 输入特征的维度
    """
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'train.csv'), parse_dates=['date'], index_col='date')
    val_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'val.csv'), parse_dates=['date'], index_col='date')
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'test.csv'), parse_dates=['date'], index_col='date')

    imf_cols = [c for c in train_df.columns if c.startswith('IMF_')]
    train_df = train_df.drop(columns=imf_cols)
    val_df = val_df.drop(columns=imf_cols)
    test_df = test_df.drop(columns=imf_cols)

    if use_external:
        with open(FEATURE_SELECTION_FILE_PATH, 'r') as f:
            selected_features = json.load(f)

        feature_cols = [DEMAND_COL] + [c for c in selected_features if c in train_df.columns]
    else:
        feature_cols = [DEMAND_COL]

    train_data = train_df[feature_cols].values
    val_data = val_df[feature_cols].values
    test_data = test_df[feature_cols].values

    x_train, y_train = create_sequences(train_data, seq_len)

    print("Sample check:")
    print("History demand (last 5 values):", x_train[0, -5:, 0])
    print("Target demand (next value):", y_train[0])

    x_val, y_val = create_sequences(val_data, seq_len)
    x_test, y_test = create_sequences(test_data, seq_len)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = x_train.shape[2]

    print(f"📊 数据加载完成 | 窗口大小: {seq_len} | 特征数: {input_dim}")
    print(f"   训练样本: {len(x_train)} | 验证样本: {len(x_val)} | 测试样本: {len(x_test)}")

    return train_loader, val_loader, test_loader, input_dim

# ======================== 3. 模型定义 =======================
class LSTMModel(nn.Module):
    """
    LSTM时间序列预测模型，包含LSTM层和全连接输出层。
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        """
        初始化LSTM模型。

        参数:
            input_dim: 输入特征的维度
            hidden_dim: LSTM隐藏层的维度
            num_layers: LSTM层数
            output_dim: 输出维度（通常为1）
            dropout: Dropout比率，用于防止过拟合
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        返回:
            预测值张量，形状为 (batch_size,)
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        return output.squeeze(-1)

# ======================== 4. 模型训练 =======================
def train_lstm(model, train_loader, val_loader, epochs: int, lr, device, patience):
    """
    训练LSTM模型，支持早停和学习率调度。

    参数:
        model: LSTM模型实例
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        epochs: 最大训练轮数
        lr: 学习率
        device: 训练设备（CPU或GPU）
        patience: 早停耐心值，验证损失不改善的最大轮数

    返回:
        model: 训练好的模型（加载最优权重）
        train_losses: 每轮的训练损失列表
        val_losses: 每轮的验证损失列表
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).squeeze(-1)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * x_batch.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            sample_pred = model(next(iter(train_loader))[0].to(device))
            print(f"Pred range: {sample_pred.min().item():.3f} ~ {sample_pred.max().item():.3f}")
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).squeeze(-1)

                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                total_val_loss += loss.item() * x_batch.size(0)

        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),models_file_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(models_file_path))
    return model, train_losses, val_losses

# ======================== 5. 模型评估 =======================
def evaluate_lstm(model, test_loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    在测试集上评估LSTM模型性能，并将归一化的预测结果还原到原始尺度。

    参数:
        model: 训练好的LSTM模型
        test_loader: 测试集数据加载器
        device: 评估设备（CPU或GPU）

    返回:
        metrics: 包含RMSE、MAE、MAPE、R²等评估指标的字典
        y_true_real: 真实值数组（原始尺度）
        y_pred_real: 预测值数组（原始尺度）
    """
    model.eval()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).squeeze(-1)

            y_pred = model(x_batch)
            y_true_list.append(y_batch.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())

    y_true = np.concatenate(y_true_list).flatten()
    y_pred = np.concatenate(y_pred_list).flatten()

    print(y_pred)
    print("y_true norm stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(y_true.min(), y_true.max(), y_true.mean()))
    print("y_pred norm stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(y_pred.min(), y_pred.max(), y_pred.mean()))
    print("Norm MSE: {:.6f}".format(np.mean((y_true - y_pred) ** 2)))

    scaled_dict = joblib.load(DEMAND_MIN_MAX_SCALER_PATH)
    d_min, d_max = scaled_dict['min'], scaled_dict['max']

    y_true_real = y_true * (d_max - d_min) + d_min
    y_pred_real = y_pred * (d_max - d_min) + d_min
    y_pred_real = np.maximum(y_pred_real, 0)

    metrics = calculate_metrics(y_true_real, y_pred_real, is_normalized=False)

    return  metrics, y_true_real, y_pred_real

# ======================== 6. 运行主函数 =======================
def run_lstm_experiment(use_external: bool=USE_EXTERNAL_FEATURES):
    """
    运行完整的LSTM基线模型实验流程，包括数据加载、模型训练、评估和结果保存。

    参数:
        use_external: 是否使用外部特征，默认为配置文件中定义的值
    """
    print("\n" + "=" * 70)
    print("🧪 LSTM 基线模型实验")
    print("=" * 70)

    train_loader, val_loader, test_loader, input_dim = load_lstm_data(seq_len=SEQ_LEN, use_external=use_external)

    print(f"📊 数据加载完成 | 窗口大小: {SEQ_LEN} | 输入维度: {input_dim}")
    print(f"🚀 使用设备: {DEVICE}")

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 模型参数: {n_params}")

    print("\n🏋️ 开始训练 LSTM...")
    start_time = time.time()

    model, train_losses, val_losses = train_lstm(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE, PATIENCE
    )

    train_time_min = (time.time() - start_time) / 60

    metrics, y_true_real, y_pred_real = evaluate_lstm(model, test_loader, DEVICE)

    print("\n📊 测试集评估结果:")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE:  {metrics['MAE']:.4f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   R²:   {metrics['R2']:.4f}")
    print(f"   训练耗时: {train_time_min:.2f} 分钟")

    results = {
        'name': 'lstm',
        'description': f'LSTM (window={SEQ_LEN}, hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, external={use_external})',
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'n_params': n_params,
        'train_time_min': train_time_min,
        'model_type': 'lstm',
        'data_mode': 'sliding_window',
        'window_size': SEQ_LEN,
        'use_external': use_external
    }
    result_df = pd.DataFrame([results])
    result_df.to_csv(result_file_path, index=False)

# ======================== 测试 =======================
if __name__ == '__main__':
    run_lstm_experiment(use_external=USE_EXTERNAL_FEATURES)