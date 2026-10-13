"""
模型训练与评估流程：单 epoch 训练、验证、指标计算和主训练循环。

支持 'vmd'（(imfs, externals) 双路输入）与 'single'（单路输入）两类模型，
并预留了端到端 DAE 的重构辅助损失接口。
"""

# ---- 引入依赖 ----
import torch
import torch.nn as nn
import numpy as np
import os
import random
import joblib
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from configs.config import (
    MODEL_FILE_PATH,                # 训练模型文件保存路径
    DEMAND_MIN_MAX_SCALER_PATH,     # 需求归一化最大最小值归一化器文件保存路径
    EARLY_STOPPING,                 # 早停开关
    PATIENCE,                       # 早停耐心值
    RECON_LOSS_WEIGHT,              # DAE重构辅助损失权重
    RANDOM_SEED,                    # 全局随机种子
)

# ---- 全局随机种子 ----
def set_seed(seed: int = RANDOM_SEED):
    """固定 Python / NumPy / PyTorch 种子并开启 cuDNN 确定性，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA卷积确定性（可能轻微降低性能，但保证可复现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---- 训练一个 epoch ----
def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        model_type: str = 'vmd',
):
    """跑完一个 epoch：前向、反向、梯度裁剪后更新参数，返回平均训练损失。

    model_type='vmd' 时按 (imfs, externals, y) 解包批次，'single' 时按 (x, y) 解包。
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    use_dae = getattr(model, 'use_dae', False)

    for batch_data in train_loader:
        if model_type == 'vmd':
            imfs, externals, y = batch_data
            imfs = imfs.to(device)
            externals = externals.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred, _ = model(imfs, externals)
        else:  # 'single'
            x, y = batch_data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred, _ = model(x)

        # 预测损失
        forecast_loss = criterion(y_pred.squeeze(-1), y.squeeze(-1))

        # DAE重构辅助损失
        dae_loss = torch.tensor(0.0, device=device)
        if use_dae and model_type == 'vmd':
            dae_loss = model.compute_dae_loss(externals)
        elif use_dae and model_type == 'single':
            # single模式：x_combined中第1列是demand，其余是外部特征
            raw_externals = x[:, :, 1:]
            dae_loss = model.compute_dae_loss(raw_externals)

        train_batch_loss = forecast_loss
        train_batch_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = imfs.size(0) if model_type == 'vmd' else x.size(0)
        # 记录总损失（预测损失+重构损失），与验证损失口径一致
        total_loss += train_batch_loss.item() * batch_size
        total_samples += batch_size

        # print("y (normalized) range:", y.min().item(), y.max().item())
        # print("y_pred (normalized) range:", y_pred.min().item(), y_pred.max().item())

    return total_loss / total_samples

# ---- 验证 ----
def validate(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        model_type: str = 'vmd',
):
    """无梯度推理整轮验证集，返回平均损失与指标字典（RMSE/MAE/MAPE/R2）。"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # 判断是否启用端到端DAE
    use_dae = getattr(model, 'use_dae', False)

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch_data in val_loader:
            if model_type == 'vmd':
                imfs, externals, y = batch_data
                imfs = imfs.to(device)
                externals = externals.to(device)
                y = y.to(device)
                y_pred, _ = model(imfs, externals)
                batch_size = imfs.size(0)
            else:  # 'single'
                x, y = batch_data
                x = x.to(device)
                y = y.to(device)
                y_pred, _ = model(x)
                batch_size = x.size(0)

            # 预测损失
            forecast_loss = criterion(y_pred.squeeze(-1), y.squeeze(-1))

            # DAE重构辅助损失
            dae_loss = torch.tensor(0.0, device=device)
            if use_dae and model_type == 'vmd':
                dae_loss = model.compute_dae_loss(externals)
            elif use_dae and model_type == 'single':
                raw_externals = x[:, :, 1:]
                dae_loss = model.compute_dae_loss(raw_externals)

            # 验证损失仅用预测损失（早停判断不受DAE重构损失影响，确保公平性）
            val_batch_loss = forecast_loss
            # 单独记录DAE损失供监控
            total_loss += val_batch_loss.item() * batch_size
            # 同时记录含重构损失的总损失（供日志对比，不影响早停）
            # total_loss += (forecast_loss + RECON_LOSS_WEIGHT * dae_loss).item() * batch_size
            total_samples += batch_size

            all_y_true.append(y.cpu())
            all_y_pred.append(y_pred.cpu())

        # 合并所有批次的预测结果和真实标签
        all_y_true = torch.cat(all_y_true, dim=0).numpy().flatten()
        all_y_pred = torch.cat(all_y_pred, dim=0).numpy().flatten()

        metrics = calculate_metrics(all_y_true, all_y_pred, is_normalized=True)

    return total_loss / total_samples, metrics

# ---- 指标计算 ----
def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_normalized: bool,
):
    """计算 RMSE / MAE / MAPE / R2，并一并返回反归一化后的真实值与预测值。

    is_normalized=True 时按 min-max 参数还原尺度，并把预测裁剪回区间内；
    MAPE 会跳过真实值为 0 的样本。
    """
    if is_normalized:
        minmax_dict = joblib.load(DEMAND_MIN_MAX_SCALER_PATH)
        d_min, d_max = minmax_dict['min'], minmax_dict['max']

        y_true = y_true * (d_max - d_min) + d_min
        y_pred = y_pred * (d_max - d_min) + d_min

        y_pred = np.clip(y_pred, d_min, d_max)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    mask = y_true != 0
    if mask.sum() == 0:
        mape = float('nan')
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # R² 决定系数
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

    return {
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2,
        'y_true_real': y_true, 'y_pred_real': y_pred,
    }

# ---- 主训练循环 ----
def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        device: str,
        weight_decay: float = 1e-5,
        model_type: str = 'vmd',
        save_path: str = None,
        early_stopping: bool = None,
        patience: int = None,
):
    """主训练循环：逐 epoch 训练+验证，按验证损失保存最优权重，支持早停。

    返回每轮记录，字段为 epoch、train_loss、val_loss、val_rmse、val_mae、val_mape、val_r2。
    """
    if save_path is None:
        save_path = MODEL_FILE_PATH

    # 早停参数：未显式传入时使用 config 默认值
    if early_stopping is None:
        early_stopping = EARLY_STOPPING
    if patience is None:
        patience = PATIENCE

    # 固定随机种子，确保训练可复现
    set_seed(RANDOM_SEED)

    # 判断是否启用端到端DAE
    use_dae = getattr(model, 'use_dae', False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 余弦退火学习率调度器：有助于跳出局部最优
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    # 针对需求预测，MSE 是最基础的 Loss
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0  # 早停计数器

    training_records = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, model_type)
        scheduler.step()  # 更新学习率

        early_stop_flag = ""
        if early_stopping:
            early_stop_flag = f" | Patience: {patience_counter}/{patience}"

        dae_info = ""
        if use_dae:
            dae_info = " [DAE端到端]"

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val RMSE: {val_metrics['RMSE']:.4f} | "
              f"Val MAE: {val_metrics['MAE']:.4f} | "
              f"Val MAPE: {val_metrics['MAPE']:.2f}% | "
              f"Val R2: {val_metrics['R2']:.4f}{early_stop_flag}{dae_info}")

        training_records.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_rmse': val_metrics['RMSE'],
            'val_mae': val_metrics['MAE'],
            'val_mape': val_metrics['MAPE'],
            'val_r2': val_metrics['R2'],
        })

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # 重置计数器
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("  ✅ Best model saved!")
        else:
            patience_counter += 1

            # 早停判断
        if early_stopping and patience_counter >= patience:
            print(f"\n⏹ 早停触发: 验证损失连续 {patience} 轮未改善，停止训练 (第 {epoch + 1} 轮)")
            break

    return training_records