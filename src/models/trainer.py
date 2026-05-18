"""
文件位置: src/models/trainer.py
作用: 模型训练与评估流程管理
      提供完整的训练循环，支持两种模型类型：
        - 'vmd': 输入为 (imfs, externals) 的VMD分解模型
        - 'single': 输入为 (x) 的单路输入模型
      包括单epoch训练、验证集评估、指标计算（RMSE/MAE/MAPE），
      以及主训练循环（最优模型保存、训练记录维护）。
      支持端到端DAE训练的辅助重构损失：当模型包含DAE编码器时，
      总损失 = 预测损失 + λ × 重构损失。
所包含的函数:
  - train_one_epoch(): 执行一个训练epoch的前向、反向传播与梯度裁剪
  - validate(): 在验证集上评估模型，计算损失与多维度指标
  - calculate_metrics(): 根据是否归一化反算真实尺度，返回RMSE/MAE/MAPE
  - train_model(): 主训练循环，调度训练/验证，保存最优模型，打印并记录性能
"""

# ======================== 0. 引入包 =======================
import torch
import torch.nn as nn
import numpy as np
import os
import joblib
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from configs.config import (
    MODEL_FILE_PATH,                # 训练模型文件保存路径
    DEMAND_MIN_MAX_SCALER_PATH,     # 需求归一化最大最小值归一化器文件保存路径
    EARLY_STOPPING,                 # 早停开关
    PATIENCE,                       # 早停耐心值
    RECON_LOSS_WEIGHT,              # DAE重构辅助损失权重
)

# ======================== 1. 训练一轮 =======================
def train_one_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        model_type: str = 'vmd',
):
    """
    执行一个完整的训练epoch

    对训练数据集中的所有批次进行前向传播、损失计算、反向传播和参数更新。
    使用梯度裁剪防止梯度爆炸，确保训练稳定性。
    支持两种模型类型的forward调用方式。

    Args:
        model (nn.Module): 待训练的深度学习模型
        train_loader (DataLoader): 训练数据加载器，提供批次数据
        optimizer (torch.optim.Optimizer): 优化器实例
        criterion (nn.Module): 损失函数
        device (torch.device): 计算设备（CPU或GPU）
        model_type (str): 模型类型，'vmd' 解包 (imfs, externals, y)，'single' 解包 (x, y)

    Returns:
        float: 当前epoch的平均训练损失
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

        # 总损失 = 预测损失 + λ × 重构损失
        total_batch_loss = forecast_loss + RECON_LOSS_WEIGHT * dae_loss
        total_batch_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = imfs.size(0) if model_type == 'vmd' else x.size(0)
        total_loss += forecast_loss.item() * batch_size  # 只记录预测损失，便于对比
        total_samples += batch_size

    return total_loss / total_samples


# ======================== 2. 验证 =======================
def validate(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        model_type: str = 'vmd',
):
    """
    在验证集上评估模型性能

    使用无梯度模式进行推理，计算验证集上的损失值，并收集所有预测结果和真实标签，
    调用calculate_metrics函数计算多维度评估指标（RMSE、MAE、MAPE）。

    Args:
        model (nn.Module): 待评估的模型
        val_loader (DataLoader): 验证数据加载器
        criterion (nn.Module): 损失函数
        device (torch.device): 计算设备
        model_type (str): 模型类型，'vmd' 或 'single'

    Returns:
        tuple:
            - float: 验证集上的平均损失值
            - dict: 包含'RMSE'、'MAE'、'MAPE'三个键的评估指标字典
    """
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

            # 验证损失含重构项（用于早停判断）
            val_batch_loss = forecast_loss + RECON_LOSS_WEIGHT * dae_loss
            total_loss += val_batch_loss.item() * batch_size
            total_samples += batch_size

            all_y_true.append(y.cpu())
            all_y_pred.append(y_pred.cpu())

        # 合并所有批次的预测结果和真实标签
        all_y_true = torch.cat(all_y_true, dim=0).numpy().flatten()
        all_y_pred = torch.cat(all_y_pred, dim=0).numpy().flatten()

        metrics = calculate_metrics(all_y_true, all_y_pred, is_normalized=True)

    return total_loss / total_samples, metrics


# ======================== 3. 计算指标 =======================
def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_normalized: bool,
):
    """
    计算回归预测的多维度评估指标

    支持归一化数据的反归一化处理，计算RMSE（均方根误差）、MAE（平均绝对误差）
    和MAPE（平均绝对百分比误差）。对于MAPE，自动排除真实值为0的样本以避免除零错误。

    Args:
        y_true (np.ndarray): 真实标签数组，形状为 (n_samples,)
        y_pred (np.ndarray): 预测值数组，形状为 (n_samples,)
        is_normalized (bool): 是否为归一化数据。若为True，则从配置文件加载
                             归一化参数并还原到原始尺度，同时确保预测值非负

    Returns:
        dict: 包含三个评估指标的字典
              - 'RMSE' (float): 均方根误差，对异常值敏感
              - 'MAE' (float): 平均绝对误差，鲁棒性强
              - 'MAPE' (float): 平均绝对百分比误差（%），当所有真实值为0时返回NaN
    """
    if is_normalized:
        minmax_dict = joblib.load(DEMAND_MIN_MAX_SCALER_PATH)
        d_min, d_max = minmax_dict['min'], minmax_dict['max']

        y_true = y_true * (d_max - d_min) + d_min
        y_pred = y_pred * (d_max - d_min) + d_min

        y_pred = np.maximum(y_pred, 0)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    mask = y_true != 0
    if mask.sum() == 0:
        mape = float('nan')
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


# ======================== 4. 主训练循环 =======================
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
    """
    执行完整的模型训练流程

    整合训练和验证过程，在每个epoch结束后进行验证集评估，打印详细的性能指标。
    采用最优模型保存策略（基于验证损失），自动保存表现最佳的模型权重。
    支持VMD分解模型和单路输入模型两种类型。
    支持端到端DAE训练的辅助重构损失。

    Args:
        model (nn.Module): 待训练的模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        epochs (int): 总训练轮数
        lr (float): 学习率
        device (torch.device): 计算设备
        weight_decay (float): L2正则化权重衰减系数，默认1e-5
        model_type (str): 模型类型，'vmd' 或 'single'，决定forward调用方式
        save_path (str): 最优模型保存路径，默认使用MODEL_FILE_PATH
        early_stopping (bool): 是否启用早停，None则使用config默认值
        patience (int): 早停耐心值，None则使用config默认值

    Returns:
        list[dict]: 训练记录列表，每个元素为一个字典，包含:
                   - 'epoch': 轮次编号
                   - 'train_loss': 训练损失
                   - 'val_loss': 验证损失
                   - 'val_rmse': 验证RMSE
                   - 'val_mae': 验证MAE
                   - 'val_mape': 验证MAPE
    """
    if save_path is None:
        save_path = MODEL_FILE_PATH

    # 早停参数：未显式传入时使用 config 默认值
    if early_stopping is None:
        early_stopping = EARLY_STOPPING
    if patience is None:
        patience = PATIENCE

    # 判断是否启用端到端DAE
    use_dae = getattr(model, 'use_dae', False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 针对需求预测，MSE 是最基础的 Loss
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0  # 早停计数器

    training_records = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, model_type)

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
              f"Val MAPE: {val_metrics['MAPE']:.2f}%{early_stop_flag}{dae_info}")

        training_records.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_rmse': val_metrics['RMSE'],
            'val_mae': val_metrics['MAE'],
            'val_mape': val_metrics['MAPE']
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