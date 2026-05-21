"""
文件位置: experiments/baselines/lstm_fair_model.py
作用: 与 XGBoost 特征平等的 LSTM/BiLSTM 对比实验 — 可复现版 + 防过拟合
"""

import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models.trainer import calculate_metrics
from src.data.preprocess import load_raw_data, split_by_time, clean_data_train, clean_data_with_stats
from experiments.baselines.xgboost_model import build_features  # 复用 XGBoost 特征工程
from configs.config import (
    SIM_DATASET_PATH, DEMAND_COL, TRAIN_RATIO, VAL_RATIO,
    BATCH_SIZE, RANDOM_SEED, MODELS_DIR, DATA_FIGURES_DIR, FIGURES_DIR
)

# ==================== 0. 全局随机种子固定（确保可复现）====================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 全局随机种子固定: seed={seed}")
    print(f"   CUDA deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   CUDA benchmark: {torch.backends.cudnn.benchmark}")

def get_worker_init_fn(seed=42):
    """DataLoader worker 进程随机种子固定"""
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
    return worker_init_fn

# ==================== 1. 公平数据加载（特征筛选 + 严格时序）====================
def load_fair_baseline_data(seq_len=12, use_feature_selection=True, importance_threshold=0.01):
    """
    加载数据，特征与 XGBoost 完全对齐，可选特征重要性筛选。
    返回字典包含 dataloaders、y 统计量、input_dim 等。
    """
    print("📂 加载数据（LSTM/BiLSTM 公平对比版 — 特征筛选+防泄露）...")
    df = load_raw_data(SIM_DATASET_PATH)
    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    # 外部特征列表（与 XGBoost 一致）
    external_cols = [
        'temperature', 'extreme_weather_days', 'wind_speed',
        'macro_invest_growth', 'gdp_growth', 'copper_price_index',
        'power_load_MW', 'equipment_aging_rate', 'bidding_intensity',
        'renewable_capacity_MW', 'is_spring_festival'
    ]

    # 分别构建特征（避免跨集合统计泄露）
    train_feat = build_features(train_clean, DEMAND_COL, external_cols, seq_len)
    val_feat = build_features(val_clean, DEMAND_COL, external_cols, seq_len)
    test_feat = build_features(test_clean, DEMAND_COL, external_cols, seq_len)

    # 获取共同特征列（排除 target）
    common_cols = list(set(train_feat.columns) & set(val_feat.columns) & set(test_feat.columns))
    common_cols = [c for c in common_cols if c != 'target']

    # 【新增】特征重要性筛选
    if use_feature_selection:
        imp_path = os.path.join(DATA_FIGURES_DIR, 'xgboost_feature_importance.csv')
        if os.path.exists(imp_path):
            imp_df = pd.read_csv(imp_path)
            top_features = imp_df[imp_df['importance'] > importance_threshold]['feature'].tolist()
            common_cols = [c for c in common_cols if c in top_features]
            print(f"   ✅ 特征筛选后保留 {len(common_cols)} 个特征（重要性 > {importance_threshold})")
        else:
            print("   ⚠️ 未找到特征重要性文件，跳过筛选")

    train_feat = train_feat[common_cols + ['target']].dropna()
    val_feat = val_feat[common_cols + ['target']].dropna()
    test_feat = test_feat[common_cols + ['target']].dropna()

    # 标准化特征（仅训练集 fit）
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(train_feat[common_cols].values)
    X_val = scaler_X.transform(val_feat[common_cols].values)
    X_test = scaler_X.transform(test_feat[common_cols].values)

    # 标签 Z-score 归一化（仅训练集统计量）
    y_train = train_feat['target'].values
    y_val = val_feat['target'].values
    y_test = test_feat['target'].values
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std
    y_test_z = (y_test - y_mean) / y_std

    # 【严格时序】在每个集合内部独立构造滑动窗口样本（无跨集合泄露）
    def create_seq(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_train_seq, y_train_seq = create_seq(X_train, y_train_z, seq_len)
    X_val_seq, y_val_seq = create_seq(X_val, y_val_z, seq_len)
    X_test_seq, y_test_seq = create_seq(X_test, y_test_z, seq_len)

    print(f"   特征维度: {len(common_cols)} (与 XGBoost 对齐)")
    print(f"   训练样本: {len(X_train_seq)} | 验证: {len(X_val_seq)} | 测试: {len(X_test_seq)}")

    worker_fn = get_worker_init_fn(seed=RANDOM_SEED)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=0, worker_init_fn=worker_fn
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test_seq)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_dim": len(common_cols),
        "seq_len": seq_len,
        "n_train": len(X_train_seq),
        "n_val": len(X_val_seq),
        "n_test": len(X_test_seq),
    }

# ==================== 2. 模型定义（低容量 + 强正则化）====================
class FairLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.0 if num_layers == 1 else dropout
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        # 小权重初始化，避免初始输出过大
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


class FairBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, num_layers=1, dropout=0.5):
        # BiLSTM 隐藏单元减半，双向拼接后与 LSTM 参数量相当
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=0.0
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.ln(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ==================== 3. 训练与评估 ====================
def train_and_eval(model_class, model_name, data_dict, epochs=300, lr=5e-4, patience=20, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model_class(input_dim=data_dict["input_dim"]).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏋️ {model_name} | 参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  # 强 L2 正则
    criterion = nn.HuberLoss(delta=0.5)  # 对异常值鲁棒
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in data_dict["train_loader"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(data_dict["train_loader"].dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in data_dict["val_loader"]:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(data_dict["val_loader"].dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f" {model_name} Epoch {epoch:3d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if no_improve >= patience:
            print(f" ⏹️ 早停于 epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    # 测试集评估
    preds_z, trues_z = [], []
    with torch.no_grad():
        for xb, yb in data_dict["test_loader"]:
            xb = xb.to(device)
            preds_z.append(model(xb).cpu().numpy())
            trues_z.append(yb.numpy())
    preds_z = np.concatenate(preds_z).flatten()
    trues_z = np.concatenate(trues_z).flatten()

    preds_real = preds_z * data_dict["y_std"] + data_dict["y_mean"]
    trues_real = trues_z * data_dict["y_std"] + data_dict["y_mean"]
    preds_real = np.maximum(preds_real, 0)

    metrics = calculate_metrics(trues_real, preds_real, is_normalized=False)
    print(f"\n📊 {model_name} 测试结果:")
    print(f" RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | MAPE: {metrics['MAPE']:.2f}% | R²: {metrics['R2']:.4f}")

    return metrics, n_params

# ==================== 4. 主函数 ====================
def run_fair_experiment():
    seed_everything(seed=RANDOM_SEED)
    print("\n" + "=" * 70)
    print("🧪 LSTM/BiLSTM 公平对比实验（可复现 + 防过拟合）")
    print("=" * 70)

    data_dict = load_fair_baseline_data(
        seq_len=12,
        use_feature_selection=True,
        importance_threshold=0.01
    )

    results = []
    for ModelClass, name in [(FairLSTM, 'FairLSTM'), (FairBiLSTM, 'FairBiLSTM')]:
        metrics, n_params = train_and_eval(ModelClass, name, data_dict)
        results.append({
            "name": name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "R2": metrics["R2"],
            "n_params": n_params
        })

    df = pd.DataFrame(results)
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_FIGURES_DIR, "fair_lstm_bilstm_results.csv"), index=False)
    print("\n📋 公平对比结果:")
    print(df.to_string(index=False))

    # 保存实验日志
    log_path = os.path.join(DATA_FIGURES_DIR, "fair_lstm_experiment_log.json")
    with open(log_path, 'w') as f:
        json.dump({
            "seed": RANDOM_SEED,
            "model_params": {
                "hidden_lstm": 16, "layers_lstm": 1,
                "hidden_bilstm": 12, "dropout": 0.5,
                "learning_rate": 5e-4, "weight_decay": 1e-2,
                "patience": 20, "loss": "Huber(delta=0.5)"
            },
            "results": results
        }, f, indent=2)
    return df

if __name__ == "__main__":
    run_fair_experiment()