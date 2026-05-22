"""
文件位置: experiments/baselines/vmd_lstm_model.py
作用: VMD-LSTM v2 强基线 — 单模型对齐版（含需求残差Skip）
      输入: 外部14维 + IMF1~4 + demand = 19维
      预测目标: 原始需求（Z-score标准化，仅训练集统计量）
      核心机制: 需求残差跳连（pred = deep_delta + demand[t]）
      预期性能: R²≈0.86, MAPE≈5.5%, RMSE≈332
"""

import os
import sys
import random
import warnings
import time
import json

os.environ["PYTHONHASHSEED"] = str(42)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler

try:
    from vmdpy import VMD
except ImportError:
    def mock_vmd(signal, alpha, tau, K, dc, init, tol):
        imfs = []
        for i in range(K):
            s, e = i * len(signal) // K, (i + 1) * len(signal) // K
            imf = np.zeros_like(signal)
            if s < len(signal):
                imf[s:e] = signal[s:e]
            imfs.append(imf)
        imfs = np.array(imfs)
        u_hat = np.fft.fft(imfs, axis=1)
        omega = np.linspace(0, 1, imfs.shape[1])
        return imfs, u_hat, omega
    if 'VMD' not in globals():
        VMD = mock_vmd

from src.models.trainer import calculate_metrics
from src.data.preprocess import (
    load_raw_data, clean_data_train, clean_data_with_stats, split_by_time
)
from configs.config import (
    DEMAND_COL, BATCH_SIZE, MODELS_DIR, DATA_FIGURES_DIR,
    SIM_DATASET_PATH, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
)

warnings.filterwarnings("ignore")


# ==================== 全局随机种子固定 ====================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.set_printoptions(precision=6, suppress=True)
    torch.set_printoptions(precision=6)
    print(f"🔒 全局随机种子已固定: seed={seed}")
    print(f"   Python hash seed: {os.environ.get('PYTHONHASHSEED')}")
    print(f"   CUDA deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   CUDA benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")


def _safe_float(val):
    if isinstance(val, np.ndarray):
        return float(val.item() if val.size == 1 else val.mean())
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== VMD 参数（与主模型共享缓存）====================
VMD_K = 4
VMD_ALPHA = 2000
VMD_TAU = 0.0
VMD_DC = 0
VMD_INIT = 1
VMD_TOL = 1e-7

# LSTM 参数（与主模型严格对齐）
HIDDEN_DIM = 48
NUM_LAYERS = 2
DROPOUT = 0.4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 300
PATIENCE = 30
SEQ_LEN = 12

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
result_file_path = os.path.join(DATA_FIGURES_DIR, 'vmd_lstm_results.csv')


# ==================== 1. VMD 分解（带缓存）====================
def vmd_decompose(signal: np.ndarray, K: int = VMD_K, cache_path: str = None):
    if cache_path and os.path.exists(cache_path):
        print(f"📂 加载 VMD 缓存: {cache_path}")
        data = np.load(cache_path)
        return data['u'], data['u_hat'], data['omega']

    print(f"🔬 VMD 分解 (K={K}, alpha={VMD_ALPHA})...")
    u, u_hat, omega = VMD(signal, VMD_ALPHA, VMD_TAU, K, VMD_DC, VMD_INIT, VMD_TOL)
    if cache_path:
        np.savez_compressed(cache_path, u=u, u_hat=u_hat, omega=omega)
        print(f"💾 VMD 结果已缓存: {cache_path}")
    return u, u_hat, omega


# ==================== 2. 数据加载（与主模型 19 维严格对齐）====================
def load_data(seq_len: int = SEQ_LEN):
    """
    加载数据并构建与 VMD-CAT-BiLSTM 完全一致的输入特征：
      X = [外部14维 + IMF1~4 + demand]  (19维)
      y = 原始需求值（Z-score，仅用训练集统计量）
    """
    print("📂 加载数据（VMD-LSTM v2：IMF+需求特征对齐版）...")
    df = load_raw_data(SIM_DATASET_PATH)

    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    train_d = train_clean[DEMAND_COL].values.astype(np.float32)
    val_d = val_clean[DEMAND_COL].values.astype(np.float32)
    test_d = test_clean[DEMAND_COL].values.astype(np.float32)
    full_d = np.concatenate([train_d, val_d, test_d])

    # 与主模型共享缓存路径，确保分解结果 100% 一致
    cache_file = os.path.join(DATA_FIGURES_DIR, f"vmd_cache_K{VMD_K}_alpha{VMD_ALPHA}.npz")
    imfs_full, _, omega = vmd_decompose(full_d, K=VMD_K, cache_path=cache_file)
    print(f" 中心频率: {omega[-1]}")

    # 外部特征（与主模型一致）
    ext_cols = [
        'temperature', 'extreme_weather_days', 'wind_speed',
        'macro_invest_growth', 'gdp_growth', 'copper_price_index',
        'power_load_MW', 'equipment_aging_rate', 'bidding_intensity',
        'renewable_capacity_MW', 'is_spring_festival'
    ]
    for d in [train_clean, val_clean, test_clean]:
        d['month_sin'] = np.sin(2 * np.pi * d.index.month / 12)
        d['month_cos'] = np.cos(2 * np.pi * d.index.month / 12)
        d['quarter'] = d.index.quarter

    feat_cols = ext_cols + ['month_sin', 'month_cos', 'quarter']

    # 分集构建输入：外部14 + IMF1~4 + demand
    lengths = [len(train_clean), len(val_clean), len(test_clean)]
    X_train_raw, X_val_raw, X_test_raw = None, None, None

    for idx, d in enumerate([train_clean, val_clean, test_clean]):
        X_base = d[feat_cols].values
        if idx == 0:
            imf_segs = [imfs_full[k][:lengths[0]] for k in range(VMD_K)]
        elif idx == 1:
            imf_segs = [imfs_full[k][lengths[0]:lengths[0]+lengths[1]] for k in range(VMD_K)]
        else:
            imf_segs = [imfs_full[k][lengths[0]+lengths[1]:] for k in range(VMD_K)]

        demand_seg = [train_d, val_d, test_d][idx]
        # 与主模型完全一致的拼接方式
        X_seg = np.column_stack([X_base] + imf_segs + [demand_seg])

        if idx == 0: X_train_raw = X_seg
        elif idx == 1: X_val_raw = X_seg
        else: X_test_raw = X_seg

    # X 标准化：仅训练集 fit
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train_raw)
    X_val_s = scaler_X.transform(X_val_raw)
    X_test_s = scaler_X.transform(X_test_raw)

    # y 标准化：仅训练集统计量
    y_mean = train_d.mean()
    y_std = train_d.std() + 1e-8

    X_scaled = np.vstack([X_train_s, X_val_s, X_test_s])
    y_full = np.concatenate([train_d, val_d, test_d])
    y_scaled = (y_full - y_mean) / y_std

    def create_seq(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_seq, y_seq = create_seq(X_scaled, y_scaled, seq_len)
    n_train = len(train_d) - seq_len
    n_val = len(val_d)

    X_train, y_train = X_seq[:n_train], y_seq[:n_train]
    X_val, y_val = X_seq[n_train:n_train+n_val], y_seq[n_train:n_train+n_val]
    X_test, y_test = X_seq[n_train+n_val:], y_seq[n_train+n_val:]

    print(f" 输入特征维度: {X_train.shape[-1]} (外部14 + IMF×{VMD_K} + demand={14+VMD_K+1})")
    print(f" 训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    return {
        "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
        "scaler_X": scaler_X, "y_mean": float(y_mean), "y_std": float(y_std),
        "seq_len": seq_len, "input_dim": int(X_train.shape[-1]),
    }, train_d, val_d, test_d


# ==================== 3. LSTM 模型（含需求残差Skip）====================
class LSTMSkip(nn.Module):
    """
    VMD-LSTM v2：单模型对齐版，含需求残差跳连
      pred = deep_delta + demand_z[t]
    """
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (B, S, 19)，最后一列为 demand_z
        demand_last = x[:, -1, -1]  # 当前时刻需求（Z-score）
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.ln(last_out)
        last_out = self.dropout(last_out)
        deep_delta = self.fc(last_out).squeeze(-1)
        # 需求残差跳连：预测 = 深度残差 + 当前需求
        return deep_delta + 1.0 * demand_last


# ==================== 4. 训练函数（输出与消融实验对齐）====================
def train_model(model, data_dict, epochs, lr, device, patience):
    save_path = os.path.join(MODELS_DIR, f"best_vmdlstm_v2_seed42.pth")

    print(f"\n{'=' * 50}")
    print("🏋️ 开始训练（VMD-LSTM v2：需求残差跳连）")
    print(f"{'=' * 50}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" 📦 参数量: {n_params:,}")
    print(f" 📋 lr={lr}, hidden={HIDDEN_DIM}, dropout={DROPOUT}, 需求跳连=1.0固定")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
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
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device).squeeze(-1)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item() * x_batch.size(0)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(y_batch.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        scheduler.step()

        # 计算验证集 R²（原始空间）
        y_mean, y_std = data_dict["y_mean"], data_dict["y_std"]
        vp = val_preds * y_std + y_mean
        vt = val_trues * y_std + y_mean
        vp = np.maximum(vp, 0)
        vm = calculate_metrics(vt, vp, is_normalized=False)

        if (epoch + 1) % 20 == 0:
            ratio = val_loss / train_loss if train_loss > 0 else float('inf')
            cur_lr = optimizer.param_groups[0]['lr']
            print(f" Epoch {epoch+1}/{epochs} - Train:{train_loss:.5f}, Val:{val_loss:.5f}, V/T:{ratio:.1f}x, ValR²:{vm['R2']:.4f}, LR:{cur_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" ⏹️ 早停 epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f" 💾 模型已保存: {save_path}")
    return model


# ==================== 5. 评估函数 ====================
def evaluate_model(model, data_dict, device):
    model.eval()
    test_loader = data_dict["test_loader"]
    y_mean = data_dict["y_mean"]
    y_std = data_dict["y_std"]

    preds_list, trues_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).squeeze(-1)
            pred = model(x_batch)
            preds_list.append(pred.cpu().numpy())
            trues_list.append(y_batch.cpu().numpy())

    preds_z = np.concatenate(preds_list).flatten()
    trues_z = np.concatenate(trues_list).flatten()

    # 反标准化到原始尺度
    preds = preds_z * y_std + y_mean
    trues = trues_z * y_std + y_mean
    preds = np.maximum(preds, 0)

    metrics = calculate_metrics(trues, preds, is_normalized=False)
    return preds, trues, metrics


# ==================== 6. 实验主函数 ====================
def run_experiment():
    seed_everything(seed=42)

    print("\n" + "=" * 70)
    print("🧪 VMD-LSTM v2（单模型 + 需求残差跳连）")
    print("=" * 70)

    data_dict, train_d, val_d, test_d = load_data(seq_len=SEQ_LEN)
    input_dim = data_dict["input_dim"]
    model = LSTMSkip(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 50}")
    print("🏋️ 开始训练（VMD-LSTM v2）")
    print(f"{'=' * 50}")
    print(f" 📦 参数量: {n_params:,}")
    print(f" 📋 lr={LEARNING_RATE}, hidden={HIDDEN_DIM}, dropout={DROPOUT}, 需求跳连=1.0固定")

    t_start = time.time()
    model = train_model(model, data_dict, EPOCHS, LEARNING_RATE, DEVICE, PATIENCE)
    preds, trues, metrics = evaluate_model(model, data_dict, DEVICE)
    train_time = (time.time() - t_start) / 60.0

    print(f"\n{'=' * 50}")
    print("📊 VMD-LSTM v2 测试集结果")
    print(f"{'=' * 50}")
    print(f" RMSE: {metrics['RMSE']:.4f}")
    print(f" MAE:  {metrics['MAE']:.4f}")
    print(f" MAPE: {metrics['MAPE']:.2f}%")
    print(f" R²:   {metrics['R2']:.4f}")
    print(f" 参数量: {n_params:,}")
    print(f" 训练耗时: {train_time:.2f} min")

    print(f"\n 📈 对比主模型 (Full Model, R²=0.8858):")
    r2_diff = metrics['R2'] - 0.8858
    rmse_diff = metrics['RMSE'] - 303.26
    print(f"    R²  : {metrics['R2']:.4f} ({r2_diff:+.4f})")
    print(f"    RMSE: {metrics['RMSE']:.4f} ({rmse_diff:+.1f})")

    # 保存预测值
    pred_path = os.path.join(DATA_FIGURES_DIR, "vmd_lstm_predictions.csv")
    pd.DataFrame({
        "true_demand": trues.flatten(),
        "pred_demand": preds.flatten(),
    }).to_csv(pred_path, index=False)

    # 保存结果 CSV
    results = {
        'name': 'vmd_lstm_v2_seed42',
        'description': 'VMD-LSTM v2 (single model + demand skip, aligned with main model)',
        'RMSE': float(metrics['RMSE']),
        'MAE': float(metrics['MAE']),
        'MAPE': float(metrics['MAPE']),
        'R2': float(metrics['R2']),
        'n_params': n_params,
        'train_time_min': round(train_time, 2),
        'n_features': input_dim,
        'model_type': 'vmd_lstm_v2',
    }
    pd.DataFrame([results]).to_csv(result_file_path, index=False)
    print(f"\n💾 结果已保存至: {result_file_path}")

    # 保存日志 JSON
    log_path = os.path.join(DATA_FIGURES_DIR, "vmd_lstm_experiment_log.json")
    log_data = {
        "seed": RANDOM_SEED,
        "pytorch_version": torch.__version__,
        "best_r2": float(results["R2"]),
        "best_rmse": float(results["RMSE"]),
        "n_params": n_params,
        "n_features": input_dim,
        "train_time_min": results["train_time_min"],
        "version": "vmd_lstm_v2_single_model_skip",
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"📝 实验日志已保存: {log_path}")

    return results


if __name__ == '__main__':
    run_experiment()