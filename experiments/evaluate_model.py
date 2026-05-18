"""
文件位置: experiments/evaluate_model.py
作用: 模型测试集评估工具
      通用评估函数，支持 VMD 双路输入模型 ('vmd') 和单路输入模型 ('single')。
      自动执行反归一化（从 Min-Max 归一化空间还原到原始需求尺度），
      计算 RMSE / MAE / MAPE 并返回真实值和预测值数组，供后续绘图与分析。
所包含的函数:
  - evaluate_model(): 在测试集上评估模型，返回指标、真实值、预测值及参数量
是否需要创建: 是，本文件需要创建（P3 实验验证专用）
相关文件更新:
  - 依赖 src/models/trainer.py 中的 calculate_metrics 函数
  - 读取 configs.config 中定义的归一化参数文件 DEMAND_MIN_MAX_SCALER_PATH
  - 接收 DataLoader 提供的批次数据（格式取决于 model_type）
  - 输出包含评估指标与预测序列的字典，用于 compare_baselines.py / plot_results.py 等脚本
"""

# ======================== 0. 引入包 =======================
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.trainer import calculate_metrics
from configs.config import DEMAND_MIN_MAX_SCALER_PATH

# ======================= 1. 评估模型 =======================
def evaluate_model(
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: str,
        model_type: str = 'vmd',
) -> dict:
    """
    在测试集上评估模型性能

    通用评测函数，支持两种模型类型。自动进行反归一化，
    返回真实尺度的评估指标和预测结果。

    Args:
        model (torch.nn.Module): 已训练好的模型
        test_loader (DataLoader): 测试数据加载器
        device (str): 计算设备
        model_type (str): 模型类型，'vmd' 解包 (imfs, externals, y)，'single' 解包 (x, y)

    Returns:
        dict: 包含以下键的评估结果字典:
            - 'RMSE' (float): 均方根误差（反归一化后）
            - 'MAE' (float): 平均绝对误差（反归一化后）
            - 'MAPE' (float): 平均绝对百分比误差（反归一化后）
            - 'y_true' (np.ndarray): 真实值（反归一化后）
            - 'y_pred' (np.ndarray): 预测值（反归一化后）
            - 'n_params' (int): 模型可训练参数总量
    """
    model.eval()
    model.to(device)

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch_data in test_loader:
            if model_type == 'vmd':
                imfs, externals, y = batch_data
                imfs = imfs.to(device)
                externals = externals.to(device)
                y = y.to(device)
                y_pred, _ = model(imfs, externals)
            else:  # 'single'
                x, y = batch_data
                x = x.to(device)
                y = y.to(device)
                y_pred, _ = model(x)

            all_y_true.append(y.cpu())
            all_y_pred.append(y_pred.cpu())

    # 合并所有批次
    all_y_true = torch.cat(all_y_true, dim=0).numpy().flatten()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy().flatten()

    # 反归一化
    metrics = calculate_metrics(all_y_true, all_y_pred, is_normalized=True)

    # 获取反归一化后的真实值和预测值（用于绘图）
    import joblib
    minmax_dict = joblib.load(DEMAND_MIN_MAX_SCALER_PATH)
    d_min, d_max = minmax_dict['min'], minmax_dict['max']
    y_true_real = all_y_true * (d_max - d_min) + d_min
    y_pred_real = all_y_pred * (d_max - d_min) + d_min
    y_pred_real = np.maximum(y_pred_real, 0)

    # 统计模型参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'y_true': y_true_real,
        'y_pred': y_pred_real,
        'n_params': n_params,
    }