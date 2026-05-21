"""
文件位置: experiments/baselines/lstm_bilstm_model.py
作用: LSTM / BiLSTM 基线对比实验 — 完全可复现版

两个模型共享相同的数据加载、预处理、训练框架，仅模型结构不同：
  - LSTM:   单向 2 层 LSTM + LayerNorm + FC
  - BiLSTM: 双向 2 层 LSTM + LayerNorm + FC

输入: 原始数据（外部特征 + 时间特征），无 VMD 分解，无差分
"""

import os
import sys
import random
import json
import time
import warnings

os.environ["PYTHONHASHSEED"] = str(42)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler

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
    print(f" PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f" CUDA version: {torch.version.cuda}")
        print(f" GPU: {torch.cuda.get_device_name(0)}")


def _safe_float(val):
    if isinstance(val, np.ndarray):
        return float(val.item() if val.size == 1 else val.mean())
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练超参数
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
PATIENCE = 30
SEQ_LEN = 12

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FIGURES_DIR, exist_ok=True)


# ==================== 1. 数据加载（修复：加入 DEMAND_COL）====================
def load_baseline_data(seq_len=SEQ_LEN):
    """
    加载原始数据，构造监督学习序列。

    【关键修复】加入 DEMAND_COL 作为输入特征，对齐 Diff-LSTM / VMD-LSTM。

    输入特征: 外部特征(11) + month_sin + month_cos + quarter + demand值 = 15维
              滑动窗口 seq_len=12 → 模型能看到 demand[t-11] ~ demand[t]
    标签:     原始需求（Z-score，仅用训练集统计量）
    """
    print("📂 加载数据（LSTM/BiLSTM 基线模式 — 含demand历史）...")
    df = load_raw_data(SIM_DATASET_PATH)
    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    # 外部特征
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
    feature_cols = external_cols + ['month_sin', 'month_cos', 'quarter']

    # 【关键修复】用 np.column_stack 显式拼接 demand 列
    # 与 Diff-LSTM / VMD-LSTM 的写法一致，避免 DataFrame 列选择问题
    train_X = np.column_stack([train_clean[feature_cols].values,
                               train_clean[DEMAND_COL].values])
    val_X = np.column_stack([val_clean[feature_cols].values,
                             val_clean[DEMAND_COL].values])
    test_X = np.column_stack([test_clean[feature_cols].values,
                              test_clean[DEMAND_COL].values])

    train_y = train_clean[DEMAND_COL].values
    val_y = val_clean[DEMAND_COL].values
    test_y = test_clean[DEMAND_COL].values

    # 拼接全集用于序列构建
    X_full = np.vstack([train_X, val_X, test_X])
    y_full = np.concatenate([train_y, val_y, test_y])

    # X 标准化（仅训练集 fit，对齐 vmd_lstm_model）
    scaler_X = StandardScaler()
    scaler_X.fit(train_X)
    X_scaled = scaler_X.transform(X_full)

    # y 标准化（仅训练集统计量）
    y_mean = train_y.mean()
    y_std = train_y.std() + 1e-8
    y_scaled = (y_full - y_mean) / y_std

    # 构建滑动窗口序列
    def create_seq(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_seq, y_seq = create_seq(X_scaled, y_scaled, seq_len)

    # 按索引切分
    n_train = len(train_X) - seq_len
    n_val = len(val_X)

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train:n_train + n_val]
    y_val = y_seq[n_train:n_train + n_val]
    X_test = X_seq[n_train + n_val:]
    y_test = y_seq[n_train + n_val:]

    print(f" 输入特征维度: {X_train.shape[-1]} (外部14 + demand=15)")
    print(f" 训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler_X": scaler_X,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_dim": X_train.shape[-1],
        "seq_len": seq_len,
    }


# ==================== 2. 模型定义 ====================
class LSTMModel(nn.Module):
    """
    基础 LSTM 模型：单向 2 层 LSTM + LayerNorm + FC

    结构:  Input → LSTM(2层) → LayerNorm → Dropout → FC → 输出
    """

    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        # Xavier 初始化
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch, seq, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        last_out = lstm_out[:, -1, :]  # 取最后时间步
        last_out = self.ln(last_out)
        last_out = self.dropout(last_out)
        return self.fc(last_out).squeeze(-1)  # (batch,)


class BiLSTMModel(nn.Module):
    """
    双向 LSTM 模型：双向 2 层 LSTM + LayerNorm + FC

    结构:  Input → BiLSTM(2层) → 拼接最后时刻正反向 → LayerNorm → Dropout → FC → 输出
    输出维度: hidden_dim * 2（拼接正向和反向）
    """

    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch, seq, input_dim)
        lstm_out, _ = self.bilstm(x)  # (batch, seq, hidden*2)
        last_out = lstm_out[:, -1, :]  # 取最后时间步（含双向信息）
        last_out = self.ln(last_out)
        last_out = self.dropout(last_out)
        return self.fc(last_out).squeeze(-1)  # (batch,)


# ==================== 3. 训练函数 ====================
def train_model(model, data_dict, model_name, epochs=EPOCHS, lr=LEARNING_RATE,
                patience=PATIENCE, device="cuda"):
    """通用训练函数，LSTM 和 BiLSTM 共用。"""
    print(f"\n{'=' * 50}")
    print(f"🏋️ 开始训练 {model_name}")
    print(f"{'=' * 50}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" 📦 参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]
    save_path = os.path.join(MODELS_DIR, f"best_{model_name}_seed42.pth")

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).squeeze(-1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).squeeze(-1)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()

        # ---- 早停 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            ratio = val_loss / train_loss if train_loss > 0 else float('inf')
            print(f" {model_name} Epoch {epoch}/{epochs} - "
                  f"Train:{train_loss:.5f}, Val:{val_loss:.5f}, V/T:{ratio:.1f}x")

        if patience_counter >= patience:
            print(f" ⏹️ {model_name} 早停 epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # 保存模型
    torch.save({
        "model_state": best_state,
        "input_dim": data_dict["input_dim"],
        "y_mean": float(data_dict["y_mean"]),
        "y_std": float(data_dict["y_std"]),
        "n_params": n_params,
    }, save_path, _use_new_zipfile_serialization=False)
    print(f" 💾 模型已保存: {save_path}")

    return model


# ==================== 4. 评估函数 ====================
def evaluate_model(model, data_dict, model_name, device="cuda"):
    """在测试集上评估，返回反标准化后的预测值和真实值。"""
    model.eval()
    test_loader = data_dict["test_loader"]
    y_mean, y_std = data_dict["y_mean"], data_dict["y_std"]
    criterion = nn.MSELoss()

    total_loss = 0.0
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            preds_list.append(pred.cpu().numpy())
            trues_list.append(yb.cpu().numpy())

    test_loss = total_loss / len(test_loader.dataset)
    preds_z = np.concatenate(preds_list).flatten()
    trues_z = np.concatenate(trues_list).flatten()

    # 反标准化
    preds = preds_z * y_std + y_mean
    trues = trues_z * y_std + y_mean
    preds = np.maximum(preds, 0)

    metrics = calculate_metrics(trues, preds, is_normalized=False)

    print(f"\n📊 {model_name} 测试集结果:")
    print(f" RMSE: {metrics['RMSE']:.4f}")
    print(f" MAE:  {metrics['MAE']:.4f}")
    print(f" MAPE: {metrics['MAPE']:.2f}%")
    print(f" R²:   {metrics['R2']:.4f}")

    return preds, trues, metrics


# ==================== 5. 单模型实验主函数 ====================
def run_single_experiment(model_class, model_name, data_dict,
                          epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,
                          device="cuda"):
    """运行单个模型的完整实验流程。"""
    t_start = time.time()

    # 构建模型
    input_dim = data_dict["input_dim"]
    model = model_class(input_dim=input_dim)

    # 训练
    model = train_model(model, data_dict, model_name,
                        epochs=epochs, lr=lr, patience=patience, device=device)

    # 评估
    preds, trues, metrics = evaluate_model(model, data_dict, model_name, device)

    train_time_min = (time.time() - t_start) / 60.0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 保存预测结果
    pred_path = os.path.join(DATA_FIGURES_DIR, f"{model_name}_predictions.csv")
    pd.DataFrame({
        "true_demand": trues.flatten(),
        "pred_demand": preds.flatten(),
    }).to_csv(pred_path, index=False)

    results = {
        "name": f"{model_name}_seed42",
        "description": f"{model_name} (hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, seq={SEQ_LEN})",
        "RMSE": _safe_float(metrics.get("RMSE", -1)),
        "MAE": _safe_float(metrics.get("MAE", -1)),
        "MAPE": _safe_float(metrics.get("MAPE", -1)),
        "R2": _safe_float(metrics.get("R2", -1)),
        "n_params": n_params,
        "train_time_min": round(train_time_min, 2),
        "model_type": model_name,
        "data_mode": "raw_demand",
        "n_features": input_dim,
    }

    return results


# ==================== 6. 完整对比实验主函数 ====================
def run_lstm_bilstm_experiment(epochs=EPOCHS, lr=LEARNING_RATE,
                               patience=PATIENCE, device="cuda"):
    """运行 LSTM 和 BiLSTM 两个基线实验，返回对比结果。"""
    seed_everything(RANDOM_SEED)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ 使用设备: {device}")

    print("\n" + "=" * 70)
    print("🧪 LSTM / BiLSTM 基线对比实验")
    print("=" * 70)

    # 加载数据（两个模型共享同一份数据）
    data_dict = load_baseline_data(seq_len=SEQ_LEN)

    all_results = []

    # ---- 实验1: LSTM ----
    print("\n" + "🔵" * 25)
    print("实验1: LSTM（单向）")
    print("🔵" * 25)
    lstm_results = run_single_experiment(
        LSTMModel, "lstm", data_dict,
        epochs=epochs, lr=lr, patience=patience, device=device
    )
    all_results.append(lstm_results)

    # ---- 实验2: BiLSTM ----
    print("\n" + "🟢" * 25)
    print("实验2: BiLSTM（双向）")
    print("🟢" * 25)
    bilstm_results = run_single_experiment(
        BiLSTMModel, "bilstm", data_dict,
        epochs=epochs, lr=lr, patience=patience, device=device
    )
    all_results.append(bilstm_results)

    # ---- 汇总对比 ----
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(DATA_FIGURES_DIR, "lstm_bilstm_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print("📋 LSTM vs BiLSTM 对比结果")
    print(f"{'=' * 70}")
    print(f"{'模型':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10} {'R²':<10} {'参数量':<12}")
    print("-" * 66)
    for _, row in results_df.iterrows():
        print(f"{row['name']:<10} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} "
              f"{row['MAPE']:<10.2f}% {row['R2']:<10.4f} {row['n_params']:<12,}")

    print(f"\n💾 结果已保存至: {csv_path}")

    return results_df


# ==================== 7. 主入口 ====================
if __name__ == "__main__":
    seed_everything(seed=RANDOM_SEED)

    results = run_lstm_bilstm_experiment(
        epochs=300,
        lr=1e-3,
        patience=30,
    )

    if not results.empty:
        log_path = os.path.join(DATA_FIGURES_DIR, "lstm_bilstm_experiment_log.json")
        log_data = {
            "seed": RANDOM_SEED,
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "lstm_r2": float(results.iloc[0]["R2"]),
            "lstm_rmse": float(results.iloc[0]["RMSE"]),
            "bilstm_r2": float(results.iloc[1]["R2"]),
            "bilstm_rmse": float(results.iloc[1]["RMSE"]),
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"\n📝 实验日志已保存: {log_path}")
