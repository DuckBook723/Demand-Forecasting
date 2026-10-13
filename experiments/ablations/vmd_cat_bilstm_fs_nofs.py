"""
消融实验：去掉特征筛选，直接用 24 维全特征训练。

VMD、CATCN、BiLSTM、GateFusion、PredictionHead 与需求跳连都保留。
输入为外部14 + IMF4 + delta_IMF4 + delta_demand + demand，共 24 维。
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


# ---- 全局随机种子 ----
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

VMD_K = 4
VMD_ALPHA = 2000
VMD_TAU = 0.0
VMD_DC = 0
VMD_INIT = 1
VMD_TOL = 1e-7

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FIGURES_DIR, exist_ok=True)


# ---- Part 1: VMD 分解（与主模型一致） ----
def vmd_decompose(signal, K=VMD_K, alpha=VMD_ALPHA, tau=VMD_TAU,
                  dc=VMD_DC, init=VMD_INIT, tol=VMD_TOL, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"📂 加载 VMD 缓存: {cache_path}")
        data = np.load(cache_path)
        return data["u"], data["u_hat"], data["omega"]
    print(f"🔬 VMD 分解 (K={K}, alpha={alpha})...")
    u, u_hat, omega = VMD(signal, alpha, tau, K, dc, init, tol)
    if cache_path:
        np.savez_compressed(cache_path, u=u, u_hat=u_hat, omega=omega)
    return u, u_hat, omega


# ---- Part 2: 模型架构（与主模型一致） ----
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 dropout=0.3, eca_kernel_size=5):
        super().__init__()
        self.layer1 = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout),
        )
        self.net = nn.Sequential(self.layer1, self.layer2)
        self.eca = ECAAttention(eca_kernel_size)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        res = self.residual_conv(x)
        out = F.relu(out + res)
        return self.eca(out)


class CATCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=16,
                 num_layers=2, kernel_size=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layers = []
        for i in range(num_layers):
            layers.append(TemporalBlock(hidden_dim, hidden_dim, kernel_size,
                                        dilation=2 ** i, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x).transpose(1, 2)
        x = self.net(x).transpose(1, 2)
        return self.output_proj(x)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


class TABiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, bidirectional=True, dropout=0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.temporal_attention = TemporalAttention(hidden_dim * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.drop(lstm_out)
        context, weights = self.temporal_attention(lstm_out)
        return context, weights


class GateFusion(nn.Module):
    def __init__(self, local_dim=16, global_dim=64, unified_dim=32):
        super().__init__()
        self.local_proj = nn.Linear(local_dim, unified_dim)
        self.global_proj = nn.Linear(global_dim, unified_dim)
        self.gate = nn.Sequential(
            nn.Linear(unified_dim * 2, unified_dim), nn.ReLU(inplace=True),
            nn.Linear(unified_dim, unified_dim), nn.Sigmoid(),
        )

    def forward(self, local_feature, global_feature):
        if local_feature.dim() == 3:
            local_feature = local_feature.mean(dim=1)
        if global_feature.dim() == 3:
            global_feature = global_feature.mean(dim=1)
        lu = self.local_proj(local_feature)
        gu = self.global_proj(global_feature)
        gate = self.gate(torch.cat([lu, gu], dim=1))
        return gate * lu + (1 - gate) * gu


class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        layers = []
        cur = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(cur, h), nn.BatchNorm1d(h),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
            ])
            cur = h
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VMD_CAT_BiLSTM_NoFS(nn.Module):
    """
    消融版模型：去掉特征筛选，直接用 24 维全特征。

    结构与主模型一致，只是输入维度固定为 24。
    """
    def __init__(self, input_dim, catcn_hidden=32, catcn_output=16,
                 catcn_layers=2, bilstm_hidden=32, bilstm_layers=1,
                 unified_dim=32, dropout=0.3,pred_hidden_dims=None):
        super().__init__()
        if pred_hidden_dims is None:
            pred_hidden_dims = [16, 8]  # 降参：原[32, 16]
        self.catcn = CATCN(input_dim, catcn_hidden, catcn_output,
                           num_layers=catcn_layers, dropout=dropout)
        self.tabilstm = TABiLSTM(input_dim, bilstm_hidden,
                                  num_layers=bilstm_layers, dropout=dropout)
        self.fusion = GateFusion(local_dim=catcn_output,
                                  global_dim=bilstm_hidden * 2,
                                  unified_dim=unified_dim)
        self.head = PredictionHead(unified_dim, hidden_dims=pred_hidden_dims,
                                    dropout=dropout)

    def forward(self, x):
        demand_last = x[:, -1, -1]
        f_local = self.catcn(x)
        f_global, attn_weights = self.tabilstm(x)
        fused = self.fusion(f_local, f_global)
        deep_delta = self.head(fused)
        out = deep_delta + 1.0 * demand_last
        return out


# ---- Part 3: 数据加载（无特征筛选，24 维全特征） ----
def load_vmd_catbilstm_data(seq_len=12, K=VMD_K):
    """
    去除特征筛选，直接使用 24 维全特征：
    外部14 + IMF4 + delta_IMF4 + delta_demand + demand。
    """
    print("📂 加载数据（w/o Feature Selection：24维全特征）...")
    df = load_raw_data(SIM_DATASET_PATH)
    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    train_demand = train_clean[DEMAND_COL].values
    val_demand = val_clean[DEMAND_COL].values
    test_demand = test_clean[DEMAND_COL].values
    full_demand = np.concatenate([train_demand, val_demand, test_demand])

    # VMD 分解
    cache_file = os.path.join(DATA_FIGURES_DIR, f"vmd_cache_K{K}_alpha{VMD_ALPHA}.npz")
    imfs_full, _, omega = vmd_decompose(full_demand, K=K, cache_path=cache_file)
    print(f" 中心频率: {omega[-1]}")

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

    lengths = [len(train_clean), len(val_clean), len(test_clean)]

    # 构建24维全特征（含动量）
    def build_features(demand_seg, imf_segs, X_base):
        imf_cols = np.column_stack(imf_segs)
        delta_imf_cols = np.zeros_like(imf_cols)
        delta_imf_cols[1:] = imf_cols[1:] - imf_cols[:-1]
        delta_demand = np.zeros_like(demand_seg)
        delta_demand[1:] = demand_seg[1:] - demand_seg[:-1]
        X = np.column_stack([X_base, imf_cols, delta_imf_cols,
                             delta_demand, demand_seg])
        return X

    X_train_raw, X_val_raw, X_test_raw = None, None, None
    for idx, d in enumerate([train_clean, val_clean, test_clean]):
        X_base = d[feature_cols].values
        if idx == 0:
            imf_segs = [imfs_full[k][:lengths[0]] for k in range(K)]
            demand_seg = train_demand
        elif idx == 1:
            imf_segs = [imfs_full[k][lengths[0]:lengths[0]+lengths[1]] for k in range(K)]
            demand_seg = val_demand
        else:
            imf_segs = [imfs_full[k][lengths[0]+lengths[1]:] for k in range(K)]
            demand_seg = test_demand

        X_seg = build_features(demand_seg, imf_segs, X_base)
        if idx == 0:
            X_train_raw = X_seg
        elif idx == 1:
            X_val_raw = X_seg
        else:
            X_test_raw = X_seg

    # 跳过特征筛选，直接使用 24 维全特征
    print(" ⏭️ 跳过特征筛选，使用24维全特征")

    # X 标准化（仅训练集 fit）
    scaler_X = StandardScaler()
    scaler_X.fit(X_train_raw)
    X_train_scaled = scaler_X.transform(X_train_raw)
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    # y 标准化（仅训练集统计量）
    y_mean = train_demand.mean()
    y_std = train_demand.std() + 1e-8

    X_scaled = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
    y_full = np.concatenate([train_demand, val_demand, test_demand])
    y_scaled = (y_full - y_mean) / y_std

    def create_seq(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_seq, y_seq = create_seq(X_scaled, y_scaled, seq_len)

    n_train = len(train_demand) - seq_len
    n_val = len(val_demand)

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train:n_train + n_val]
    y_val = y_seq[n_train:n_train + n_val]
    X_test = X_seq[n_train + n_val:]
    y_test = y_seq[n_train + n_val:]

    print(f" 输入特征维度: {X_train.shape[-1]} "
          f"(外部14 + IMF4 + delta_IMF4 + delta_demand + demand)")
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
        "seq_len": seq_len,
        "input_dim": X_train.shape[-1],
    }, train_demand, val_demand, test_demand


# ---- Part 4: 训练（与主模型一致） ----
def train_model(model, data_dict, epochs=300, patience=30, lr=5e-4, device="cuda"):
    print(f"\n{'=' * 50}")
    print("🏋️ 开始训练（w/o Feature Selection：24维全特征）")
    print(f"{'=' * 50}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" 📦 参数量: {n_params:,}")
    print(f" 📋 lr={lr}, weight_decay=1e-3, dropout=0.3, 需求跳连=1.0固定")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]

    for epoch in range(1, epochs + 1):
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

        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).squeeze(-1)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            ratio = val_loss / train_loss if train_loss > 0 else float('inf')
            cur_lr = optimizer.param_groups[0]['lr']
            y_mean, y_std = data_dict["y_mean"], data_dict["y_std"]
            vp = val_preds * y_std + y_mean
            vt = val_trues * y_std + y_mean
            vp = np.maximum(vp, 0)
            vm = calculate_metrics(vt, vp, is_normalized=False)
            print(f" Epoch {epoch}/{epochs} - "
                  f"Train:{train_loss:.5f}, Val:{val_loss:.5f}, V/T:{ratio:.1f}x, "
                  f"ValR²:{vm['R2']:.4f}, LR:{cur_lr:.2e}")

        if no_improve >= patience:
            print(f" ⏹️ 早停 epoch {epoch}")
            break

    model.load_state_dict(best_state)
    save_path = os.path.join(MODELS_DIR, "best_vmdcatbilstm_fs_nofs_seed42.pth")
    torch.save({
        "model_state": best_state,
        "input_dim": data_dict["input_dim"],
        "y_mean": float(data_dict["y_mean"]),
        "y_std": float(data_dict["y_std"]),
        "n_params": n_params,
    }, save_path, _use_new_zipfile_serialization=False)
    print(f" 💾 模型已保存: {save_path}")
    return model


# ---- Part 5: 评估（与主模型一致） ----
def evaluate_model(model, data_dict, device="cuda"):
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

    preds_z = np.concatenate(preds_list).flatten()
    trues_z = np.concatenate(trues_list).flatten()

    preds = preds_z * y_std + y_mean
    trues = trues_z * y_std + y_mean
    preds = np.maximum(preds, 0)

    metrics = calculate_metrics(trues, preds, is_normalized=False)
    return preds, trues, metrics


# ---- Part 6: 实验主函数 ----
def run_vmd_catbilstm_experiment(epochs=300, patience=30, lr=5e-4,
                                  device="cuda") -> pd.DataFrame:
    seed_everything(RANDOM_SEED)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ 使用设备: {device}")

    print("\n" + "=" * 70)
    print("🧪 VMD-CAT-BiLSTM w/o Feature Selection（24维全特征）")
    print("=" * 70)

    t_start = time.time()
    data_dict, train_demand, val_demand, test_demand = load_vmd_catbilstm_data(seq_len=12)

    input_dim = data_dict["input_dim"]
    model = VMD_CAT_BiLSTM_NoFS(
        input_dim=input_dim,
        catcn_hidden=16,  # 与主模型一致，原32
        bilstm_hidden=24,  # 与主模型一致，原32
        unified_dim=24,  # 与主模型一致，原32
        pred_hidden_dims=[16, 8]  # 与主模型一致，原[32,16]
    )

    model = train_model(model, data_dict, epochs, patience, lr, device)
    preds, trues, metrics = evaluate_model(model, data_dict, device)

    train_time_min = (time.time() - t_start) / 60.0

    print(f"\n{'=' * 50}")
    print("📊 w/o Feature Selection 测试集结果")
    print(f"{'=' * 50}")
    print(f" RMSE: {metrics['RMSE']:.4f}")
    print(f" MAE:  {metrics['MAE']:.4f}")
    print(f" MAPE: {metrics['MAPE']:.2f}%")
    print(f" R²:   {metrics['R2']:.4f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" 参数量: {n_params:,}")
    print(f" 训练耗时: {train_time_min:.2f} min")

    print(f"\n 📈 对比主模型 (Full Model + FS, R²=0.8858):")
    r2_diff = metrics['R2'] - 0.8858
    rmse_diff = metrics['RMSE'] - 303.26
    print(f"    R²  : {metrics['R2']:.4f} ({r2_diff:+.4f})")
    print(f"    RMSE: {metrics['RMSE']:.4f} ({rmse_diff:+.1f})")

    result_path = os.path.join(DATA_FIGURES_DIR, "vmd_catbilstm_nofs_predictions.csv")
    pd.DataFrame({
        "true_demand": trues.flatten(),
        "pred_demand": preds.flatten(),
    }).to_csv(result_path, index=False)

    results = {
        "name": "vmd_cat_bilstm_fs_nofs_seed42",
        "description": "Ablation: VMD-CAT-BiLSTM w/o Feature Selection (24-dim full features)",
        "RMSE": _safe_float(metrics.get("RMSE", -1)),
        "MAE": _safe_float(metrics.get("MAE", -1)),
        "MAPE": _safe_float(metrics.get("MAPE", -1)),
        "R2": _safe_float(metrics.get("R2", -1)),
        "n_params": n_params,
        "train_time_min": round(train_time_min, 2),
        "n_features": input_dim,
        "model_type": "vmd_cat_bilstm_fs_nofs",
    }
    results_df = pd.DataFrame([results])
    csv_path = os.path.join(DATA_FIGURES_DIR, "vmd_catbilstm_fs_nofs_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\n💾 结果已保存至: {csv_path}")

    log_path = os.path.join(DATA_FIGURES_DIR, "vmd_catbilstm_fs_nofs_log.json")
    log_data = {
        "seed": RANDOM_SEED,
        "pytorch_version": torch.__version__,
        "best_r2": float(results["R2"]),
        "best_rmse": float(results["RMSE"]),
        "n_params": n_params,
        "n_features": input_dim,
        "train_time_min": results["train_time_min"],
        "version": "ablation_nofs_24dim_full_features",
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"📓 实验日志已保存: {log_path}")

    return results_df


# ---- Part 7: 主入口 ----
if __name__ == "__main__":
    seed_everything(seed=RANDOM_SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)

    results = run_vmd_catbilstm_experiment(
        epochs=300,
        patience=30,
        lr=5e-4,
    )