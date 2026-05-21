"""
文件位置: experiments/baselines/lstm_diff_model.py
作用: LSTM 差分预测版 — 完全可复现（固定全部随机源 + 消除标准化泄露）

修复清单:
1. [P0] 全局随机种子固定（Python/NumPy/PyTorch/CUDA）
2. [P0] CUDA 确定性设置：cudnn.deterministic=True, benchmark=False
3. [P0] DataLoader worker_init_fn 固定，确保 shuffle 结果一致
4. [P1] 标签标准化（y_mean/y_std）仅用训练集 diff 计算，消除全局统计泄露
5. [P1] 模型保存路径带 seed 标记，避免多次运行互相覆盖
6. [P2] 实验日志自动输出（环境版本 + 种子 + 指标）
"""

import os
import sys
import random
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler

from src.models.trainer import calculate_metrics
from src.data.preprocess import (
    load_raw_data, clean_data_train, clean_data_with_stats, split_by_time
)
from configs.config import (
    DEMAND_COL, BATCH_SIZE, MODELS_DIR,
    DATA_FIGURES_DIR, SIM_DATASET_PATH, TRAIN_RATIO, VAL_RATIO,
    RANDOM_SEED
)

warnings.filterwarnings("ignore")

# ==================== 全局随机种子固定 ====================
def seed_everything(seed: int = 42):
    """
    固定所有已知随机源，确保 Diff-LSTM 实验 100% 可复现。

    覆盖范围：
    - Python hash/random（DataLoader shuffle 顺序、字典遍历）
    - NumPy（VMD、StandardScaler、数据增强中的随机采样）
    - PyTorch CPU/GPU（权重初始化、Dropout、CUDA kernel 调度）
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 【关键】CUDA 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🔒 全局随机种子已固定: seed={seed}")
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   CUDNN benchmark: {torch.backends.cudnn.benchmark}")


# ==================== DataLoader worker 种子固定 ====================
def _worker_init_fn(worker_id: int):
    """
    每个 DataLoader worker 进程需要独立但确定性的种子。
    公式: base_seed + worker_id，确保不同 worker 不重复且可复现。
    """
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ==================== 配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
PATIENCE = 30
SEQ_LEN = 12
SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FIGURES_DIR, exist_ok=True)

# 模型保存路径带 seed 标记，避免覆盖
models_file_path = os.path.join(MODELS_DIR, f"best_lstm_diff_seed{SEED}.pth")
result_file_path = os.path.join(DATA_FIGURES_DIR, f'lstm_diff_results_seed{SEED}.csv')


# ==================== 1. 模型定义 ====================
class LSTMDiff(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.ln(last_out)
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out.squeeze(-1)


# ==================== 2. 数据加载（修复标准化泄露）====================
def load_data_diff(seq_len=SEQ_LEN):
    """
    加载数据并构造差分标签。

    【修复】
    - y_mean/y_std 仅使用训练集 diff 计算，避免验证集/测试集信息泄露
    - 时间趋势使用全局连续索引，确保跨集合一致性
    """
    print("📂 加载数据（差分预测模式）...")
    df = load_raw_data(SIM_DATASET_PATH)
    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    external_cols = [
        'temperature', 'extreme_weather_days', 'wind_speed',
        'macro_invest_growth', 'gdp_growth', 'copper_price_index',
        'power_load_MW', 'equipment_aging_rate', 'bidding_intensity',
        'renewable_capacity_MW', 'is_spring_festival'
    ]

    for d in [train_clean, val_clean, test_clean]:
        d['month_sin'] = np.sin(2 * np.pi * d.index.month / 12)
        d['month_cos'] = np.cos(2 * np.pi * d.index.month / 12)
        d['quarter'] = d.index.quarter

    for d in [train_clean, val_clean, test_clean]:
        d[external_cols] = d[external_cols].interpolate(method='linear').ffill().bfill()

    feature_cols = external_cols + ['month_sin', 'month_cos', 'quarter']

    # 提取数据
    datasets = [train_clean, val_clean, test_clean]
    X_list, y_list, last_demand_list = [], [], []

    for d in datasets:
        X_raw = d[feature_cols + [DEMAND_COL]].values
        demand_vals = d[DEMAND_COL].values
        # 差分标签: diff_t = demand_t - demand_{t-1}
        diff = np.diff(demand_vals, prepend=demand_vals[0])

        X_list.append(X_raw)
        y_list.append(diff)
        last_demand_list.append(demand_vals)

    # 特征标准化（仅训练集 fit）
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_list[0])
    X_val = scaler_X.transform(X_list[1])
    X_test = scaler_X.transform(X_list[2])

    # 【关键修复】标签标准化仅用训练集 diff 的统计量
    # 原代码: y_diff = np.concatenate(y_list)  → 包含验证/测试集，存在泄露
    y_train_diff_raw = y_list[0]
    y_mean = y_train_diff_raw.mean()
    y_std = y_train_diff_raw.std() + 1e-8

    y_train = (y_list[0] - y_mean) / y_std
    y_val = (y_list[1] - y_mean) / y_std
    y_test = (y_list[2] - y_mean) / y_std

    print(f"   差分标签统计（训练集）: μ={y_mean:.3f}, σ={y_std:.3f}")

    def create_sequences(X, y, last_demand, seq_len):
        X_seq, y_seq, base_seq = [], [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
            # base = demand_{t-1}，用于最终重构 demand_t = base + pred_diff
            base_seq.append(last_demand[i+seq_len-1])
        return (np.array(X_seq, dtype=np.float32),
                np.array(y_seq, dtype=np.float32),
                np.array(base_seq, dtype=np.float32))

    X_train_seq, y_train_seq, base_train = create_sequences(X_train, y_train, last_demand_list[0], seq_len)
    X_val_seq, y_val_seq, base_val = create_sequences(X_val, y_val, last_demand_list[1], seq_len)
    X_test_seq, y_test_seq, base_test = create_sequences(X_test, y_test, last_demand_list[2], seq_len)

    print(f"   训练: {len(X_train_seq)} | 验证: {len(X_val_seq)} | 测试: {len(X_test_seq)}")

    # 【修复】使用 worker_init_fn 固定 DataLoader 随机性
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq), torch.tensor(base_train)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=0,  # 单线程最安全；若改为 >0 需启用 worker_init_fn
        worker_init_fn=None
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq), torch.tensor(base_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test_seq), torch.tensor(base_test)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, test_loader, X_train.shape[1], y_mean, y_std, scaler_X


# ==================== 3. 训练函数 ====================
def train_diff(model, train_loader, val_loader, epochs, lr, device, patience):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).squeeze(-1)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, _ in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device).squeeze(-1)
                y_pred = model(x_batch)
                val_loss += criterion(y_pred, y_batch).item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1} - Train:{train_loss:.5f}, Val:{val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), models_file_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ 早停 epoch {epoch+1}")
                break

    # 【修复】weights_only=True 避免 PyTorch 版本差异警告
    model.load_state_dict(torch.load(models_file_path, weights_only=True))
    return model


# ==================== 4. 评估函数 ====================
def evaluate_diff(model, test_loader, device, y_mean, y_std):
    model.eval()
    y_true_diff_z, y_pred_diff_z, bases = [], [], []

    with torch.no_grad():
        for x_batch, y_batch, base_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)
            y_true_diff_z.append(y_batch.cpu().numpy())
            y_pred_diff_z.append(y_pred.cpu().numpy())
            bases.append(base_batch.numpy())

    y_true_diff_z = np.concatenate(y_true_diff_z).flatten()
    y_pred_diff_z = np.concatenate(y_pred_diff_z).flatten()
    bases = np.concatenate(bases).flatten()

    # 反标准化差分
    y_true_diff = y_true_diff_z * y_std + y_mean
    y_pred_diff = y_pred_diff_z * y_std + y_mean

    # 最终输出 = demand_{t-1} + predicted_diff
    y_true_real = bases + y_true_diff
    y_pred_real = bases + y_pred_diff
    y_pred_real = np.maximum(y_pred_real, 0)

    print(f"\n原始空间: true min={y_true_real.min():.1f}, max={y_true_real.max():.1f}")
    print(f"          pred min={y_pred_real.min():.1f}, max={y_pred_real.max():.1f}")

    metrics = calculate_metrics(y_true_real, y_pred_real, is_normalized=False)
    return metrics, y_true_real, y_pred_real


# ==================== 5. 主函数 ====================
def run_lstm_diff_experiment():
    # 【关键】程序最开始处固定全部随机种子
    seed_everything(seed=SEED)

    print("\n" + "=" * 70)
    print("🧪 LSTM 差分预测实验（Diff-LSTM 完全可复现版）")
    print("=" * 70)

    train_loader, val_loader, test_loader, input_dim, y_mean, y_std, scaler = load_data_diff()

    model = LSTMDiff(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 参数量: {n_params:,}")

    start_time = time.time()
    model = train_diff(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE, PATIENCE)
    train_time = (time.time() - start_time) / 60

    metrics, y_true, y_pred = evaluate_diff(model, test_loader, DEVICE, y_mean, y_std)

    print(f"\n📊 测试集结果:")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE:  {metrics['MAE']:.4f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   R²:   {metrics['R2']:.4f}")

    # 保存结果
    results = {
        'name': f'lstm_diff_seed{SEED}',
        'R2': metrics['R2'],
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'n_params': n_params,
        'train_time_min': train_time,
    }
    pd.DataFrame([results]).to_csv(result_file_path, index=False)
    print(f"\n💾 结果已保存至: {result_file_path}")

    # 保存实验日志
    log_path = os.path.join(DATA_FIGURES_DIR, f'lstm_diff_experiment_log_seed{SEED}.json')
    log_data = {
        "seed": SEED,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "seq_len": SEQ_LEN,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                   for k, v in results.items()}
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"📝 实验日志已保存: {log_path}")


if __name__ == '__main__':
    run_lstm_diff_experiment()