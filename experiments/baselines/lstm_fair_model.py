"""LSTM / BiLSTM 公平对比基线，使用与 XGBoost 相同的特征子集。"""

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
from experiments.baselines.xgboost_model import build_features
from configs.config import (
    SIM_DATASET_PATH, DEMAND_COL, TRAIN_RATIO, VAL_RATIO,
    BATCH_SIZE, RANDOM_SEED, DATA_FIGURES_DIR,
)

import warnings
warnings.filterwarnings("ignore")


# ---- 全局随机种子 ----
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
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   CUDA benchmark: {torch.backends.cudnn.benchmark}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ---- 核心特征子集（约 20 维）----
# 依据 XGBoost 特征重要性，只保留对月度需求预测最关键的特征
CORE_FEATURES = [
    # 原始外部特征（11 维）
    'temperature', 'extreme_weather_days', 'wind_speed',
    'macro_invest_growth', 'gdp_growth', 'copper_price_index',
    'power_load_MW', 'equipment_aging_rate', 'bidding_intensity',
    'renewable_capacity_MW', 'is_spring_festival',
    # 时间特征（2 维）
    'month', 'quarter',
    # 需求滞后（2 维）— 最关键
    'demand_lag_1', 'demand_lag_12',
    # 滚动统计（1 维）
    'demand_roll_mean_12',
    # 差分（1 维）
    'demand_diff_1',
    # 外部滞后（选 3 个与需求相关性最高的）
    'ext_macro_invest_growth_lag_1',
    'ext_power_load_MW_lag_1',
    'ext_temperature_lag_1',
]


# ---- 数据加载 ----
def load_fair_baseline_data(seq_len=12, use_feature_selection=True, importance_threshold=0.01):
    print("📂 加载数据（LSTM/BiLSTM 公平对比版 — 精简特征与 XGBoost 对齐）...")
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

    # 分别构建特征（避免跨集合统计泄露）
    train_feat = build_features(train_clean, DEMAND_COL, external_cols, seq_len)
    val_feat = build_features(val_clean, DEMAND_COL, external_cols, seq_len)
    test_feat = build_features(test_clean, DEMAND_COL, external_cols, seq_len)

    # 获取共同特征列
    all_cols = [c for c in train_feat.columns if c != 'target' and c in val_feat.columns and c in test_feat.columns]

    # 特征筛选策略
    if use_feature_selection:
        imp_path = os.path.join(DATA_FIGURES_DIR, 'xgboost_feature_importance.csv')
        if os.path.exists(imp_path):
            imp_df = pd.read_csv(imp_path)
            top_features = imp_df[imp_df['importance'] > importance_threshold]['feature'].tolist()
            common_cols = [c for c in all_cols if c in top_features]
            print(f"   ✅ 基于 XGBoost 重要性筛选，保留 {len(common_cols)} 个特征")
        else:
            # 未找到重要性文件时退回预定义核心特征子集，避免使用全部 55 维
            common_cols = [c for c in all_cols if c in CORE_FEATURES]
            print(f"   ⚠️ 未找到特征重要性文件，使用预定义核心特征子集（{len(common_cols)} 维）")
    else:
        common_cols = all_cols
        print(f"   ℹ️ 使用全部特征（{len(common_cols)} 维）")

    # 核心特征至少要有一部分存在
    if len(common_cols) == 0:
        raise ValueError("❌ 特征交集为空，请检查 CORE_FEATURES 与 build_features 输出是否匹配")

    # 先按列筛选再 dropna，去掉滞后特征带来的空值
    train_feat = train_feat[common_cols + ['target']].dropna()
    val_feat = val_feat[common_cols + ['target']].dropna()
    test_feat = test_feat[common_cols + ['target']].dropna()

    # 确认没有残留 NaN
    for name, feat_df in [("train", train_feat), ("val", val_feat), ("test", test_feat)]:
        if feat_df.isnull().any().any():
            raise ValueError(f"❌ {name}_feat 仍存在 NaN")

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

    print(f"   特征维度: {len(common_cols)} (精简后与 XGBoost 核心特征对齐)")
    print(f"   训练样本: {len(train_feat)} | 验证: {len(val_feat)} | 测试: {len(test_feat)}")

    # 各集合内部独立构造滑动窗口，避免跨集合取到未来信息
    def create_seq(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_train_seq, y_train_seq = create_seq(X_train, y_train_z, seq_len)
    X_val_seq, y_val_seq = create_seq(X_val, y_val_z, seq_len)
    X_test_seq, y_test_seq = create_seq(X_test, y_test_z, seq_len)

    print(f"   序列样本 — 训练: {len(X_train_seq)} | 验证: {len(X_val_seq)} | 测试: {len(X_test_seq)}")

    assert not np.isnan(X_train_seq).any(), "X_train_seq 包含 NaN"
    assert not np.isnan(y_train_seq).any(), "y_train_seq 包含 NaN"

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val_seq)),
        batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test_seq)),
        batch_size=BATCH_SIZE, shuffle=False)

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


# ---- 模型定义 ----
class FairLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=48, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.0 if num_layers == 1 else dropout
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


class FairBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.ln(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ---- 训练与评估 ----
def train_and_eval(model_class, model_name, data_dict, epochs=300, lr=1e-3, patience=30, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model_class(input_dim=data_dict["input_dim"]).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 50}")
    print(f"🏋️ 开始训练（{model_name}）")
    print(f"{'=' * 50}")
    print(f" 📦 参数量: {n_params:,}")
    print(f" 📋 lr={lr}, weight_decay=1e-3, dropout=0.3, loss=MSE")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in data_dict["train_loader"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(data_dict["train_loader"].dataset)

        model.eval()
        val_loss = 0.0
        val_preds_z, val_trues_z = [], []
        with torch.no_grad():
            for xb, yb in data_dict["val_loader"]:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vloss = criterion(pred, yb)
                if not torch.isnan(vloss):
                    val_loss += vloss.item() * xb.size(0)
                val_preds_z.append(pred.cpu().numpy())
                val_trues_z.append(yb.cpu().numpy())
        val_loss /= len(data_dict["val_loader"].dataset)
        val_preds_z = np.concatenate(val_preds_z)
        val_trues_z = np.concatenate(val_trues_z)
        scheduler.step(val_loss)

        # 验证集 R²（原始空间）
        vp = val_preds_z * data_dict["y_std"] + data_dict["y_mean"]
        vt = val_trues_z * data_dict["y_std"] + data_dict["y_mean"]
        vm = calculate_metrics(vt, vp, is_normalized=False)

        if epoch % 20 == 0 or epoch == 1:
            ratio = val_loss / train_loss if train_loss > 0 else float('inf')
            cur_lr = optimizer.param_groups[0]['lr']
            print(f" Epoch {epoch:3d}/{epochs} - Train:{train_loss:.5f}, Val:{val_loss:.5f}, V/T:{ratio:.1f}x, ValR²:{vm['R2']:.4f}, LR:{cur_lr:.2e}")

        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

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
    pred_csv_path = os.path.join(DATA_FIGURES_DIR, f"{model_name.lower()}_predictions.csv")
    pd.DataFrame({
        "true_demand": trues_real.flatten(),
        "pred_demand": preds_real.flatten(),
    }).to_csv(pred_csv_path, index=False)
    print(f"💾 预测曲线数据已保存: {pred_csv_path}")
    print(f"\n{'=' * 50}")
    print(f"📊 {model_name} 测试集结果")
    print(f"{'=' * 50}")
    print(f" RMSE: {metrics['RMSE']:.4f}")
    print(f" MAE:  {metrics['MAE']:.4f}")
    print(f" MAPE: {metrics['MAPE']:.2f}%")
    print(f" R²:   {metrics['R2']:.4f}")
    print(f" 参数量: {n_params:,}")

    print(f"\n 📈 对比主模型 (Full Model, R²=0.8858):")
    r2_diff = metrics['R2'] - 0.8858
    rmse_diff = metrics['RMSE'] - 303.26
    print(f"    R²  : {metrics['R2']:.4f} ({r2_diff:+.4f})")
    print(f"    RMSE: {metrics['RMSE']:.4f} ({rmse_diff:+.1f})")

    return metrics, n_params


# ---- 主函数 ----
def run_fair_experiment():
    seed_everything(seed=RANDOM_SEED)
    print("\n" + "=" * 70)
    print("🧪 LSTM/BiLSTM 公平对比实验（精简特征 + 提升容量）")
    print("=" * 70)

    data_dict = load_fair_baseline_data(
        seq_len=12,
        use_feature_selection=True,
        importance_threshold=0.01
    )

    all_results = []
    for ModelClass, name in [(FairLSTM, 'FairLSTM'), (FairBiLSTM, 'FairBiLSTM')]:
        metrics, n_params = train_and_eval(ModelClass, name, data_dict)
        all_results.append({
            "name": name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "R2": metrics["R2"],
            "n_params": n_params,
            "train_time_min": 0,
        })

    df = pd.DataFrame(all_results)
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_FIGURES_DIR, "fair_lstm_bilstm_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print("📋 公平对比实验汇总")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    print(f"\n💾 结果已保存至: {csv_path}")

    log_path = os.path.join(DATA_FIGURES_DIR, "fair_lstm_experiment_log.json")
    with open(log_path, 'w') as f:
        json.dump({
            "seed": RANDOM_SEED,
            "models": {
                "FairLSTM": {"hidden": 48, "layers": 2},
                "FairBiLSTM": {"hidden": 32, "layers": 2}
            },
            "hyperparams": {
                "lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.3,
                "patience": 30, "loss": "MSE"
            },
            "results": all_results,
        }, f, indent=2)
    print(f"📝 实验日志已保存: {log_path}")
    return df


if __name__ == "__main__":
    run_fair_experiment()