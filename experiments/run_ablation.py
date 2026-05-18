"""
文件位置: experiments/run_ablation.py
作用: 消融实验统一运行脚本
      依次训练并评估 Full 模型与 5 个消融变体（A1-A5），
      记录训练过程到 figures/data/ablation/，
      汇总测试集指标到 figures/data/ablation_results.csv。

消融实验设计:
  - Full (VMD-CAT-BiLSTM): 完整模型基线（端到端DAE）
  - A1 w/o VMD: 去掉VMD分解，直接用原始demand
  - A2 w/o CA:  去掉ECA通道注意力
  - A3 w/o TA:  去掉时间注意力
  - A4 w/o GateFusion: 门控融合替换为简单拼接
  - A5 w/o DAE: 去掉DAE编码，直接用原始5特征

端到端DAE说明:
  当 DAE_END_TO_END=True 时，Full/A1/A2/A3/A4 模型内部集成DAE编码器，
  接收原始5维特征并在模型内部编码为3维，DAE权重根据预测损失+重构损失联合优化。
  A5 w/o DAE 不包含DAE编码器，直接使用5维原始特征。

使用方式:
  python experiments/run_ablation.py
  python experiments/run_ablation.py --epochs 100 --skip_full
"""

# ======================== 0. 引入包 =======================
import os
import sys
import time
import json
import torch
import pandas as pd
from configs.config import (
    DATA_PROCESSED_DIR,                 # 处理后数据目录
    FEATURE_SELECTION_FILE_PATH,        # 特征筛选结果文件路径
    DEMAND_COL,                         # 目标列名
    SEQ_LEN,                            # 滑动窗口长度
    BATCH_SIZE,                         # 批次大小
    EPOCHS,                             # 训练轮次
    LEARNING_RATE,                      # 学习率
    WEIGHT_DECAY,                       # L2 正则化权重衰减系数
    TCN_HIDDEN,                         # TCN 隐藏层维度
    TCN_OUTPUT,                         # TCN 输出维度
    LSTM_HIDDEN,                        # LSTM隐藏层数
    LSTM_LAYERS,                        # LSTM层数
    UNIFIED_DIM,                        # 联合输出维度
    ABLATION_MODELS_DIR,                # 消融实验模型保存目录
    ABLATION_RECORDS_DIR,               # 消融实验训练记录保存目录
    ABLATION_RESULTS_PATH,              # 消融实验汇总结果文件路径
    DROPOUT,                            # dropout 概率
    EARLY_STOPPING,                     # 提前停止
    PATIENCE,                           # 提前停止的耐心轮数
    SHARE_SUBNET,                       # 共享子网络
    DAE_END_TO_END,                     # DAE端到端训练开关
    DAE_MODEL_FILE_PATH,                # DAE 模型文件路径
)
from src.data.dataset import get_dataloaders
from src.models.trainer import train_model
from src.models.vmd_cat_bilstm import VMDCATBiLSTM
from experiments.ablation.no_vmd import NoVMDModel
from experiments.ablation.no_ca import NoCAModel
from experiments.ablation.no_ta import NoTAModel
from experiments.ablation.no_gate_fusion import NoGateFusionModel
from experiments.evaluate_model import evaluate_model

# ======================== 1. 消融实验配置 =======================
def get_ablation_configs():
    """
    获取消融实验的所有配置列表。

    根据DAE_END_TO_END开关自动选择数据模式和模型参数：
    - DAE_END_TO_END=True: 所有VMD模型使用原始5特征+端到端DAE编码
    - DAE_END_TO_END=False: 使用预计算的3维编码特征

    该函数读取特征选择文件，并构建6个实验配置，包括完整模型基线和5个消融变体：
    - Full: 完整的VMD-CAT-BiLSTM模型
    - A1 (w/o VMD): 移除VMD分解
    - A2 (w/o CA): 移除通道注意力机制
    - A3 (w/o TA): 移除时间注意力机制
    - A4 (w/o GateFusion): 使用简单拼接替代门控融合
    - A5 (w/o DAE): 不使用DAE特征编码

    Returns:
        list: 包含多个字典的配置列表，每个字典包含以下键：
            - name (str): 实验名称标识
            - desc (str): 实验描述信息
            - model_class (nn.Module): 模型类
            - model_type (str): 模型类型 ('vmd' 或 'single')
            - data_mode (str): 数据模式 ('vmd_dae', 'no_vmd', 'no_dae')
            - model_kwargs (dict): 模型额外参数
            - ext_cols_override (list or None): 外部特征列覆盖设置

    Example:
        configs = get_ablation_configs()
        for config in configs:
            print(config['name'], config['desc'])
    """
    selected_features = None
    if os.path.exists(FEATURE_SELECTION_FILE_PATH):
        with open(FEATURE_SELECTION_FILE_PATH, 'r') as f:
            selected_features = json.load(f)

    if DAE_END_TO_END:
        # ===== 端到端DAE模式 =====
        # 所有VMD模型读取原始CSV（5个原始特征），模型内部DAE编码5→3
        # w/o DAE模型读取原始CSV（5个原始特征），不经过DAE编码
        configs = [
            # ---- Full 模型基线 ----
            {
                'name': 'full',
                'desc': 'VMD-CAT-BiLSTM (Full)',
                'model_class': VMDCATBiLSTM,
                'model_type': 'vmd',
                'data_mode': 'no_dae',  # 读取原始CSV，5个原始特征
                'model_kwargs': {'use_dae': True},  # 模型内部DAE编码5→3
                'ext_cols_override': selected_features,
            },
            # ---- A1: w/o VMD ----
            {
                'name': 'no_vmd',
                'desc': 'w/o VMD (无VMD分解)',
                'model_class': NoVMDModel,
                'model_type': 'single',
                'data_mode': 'no_vmd_raw',  # 读取原始CSV，demand+5原始特征
                'model_kwargs': {'use_dae': True},  # 模型内部DAE编码5→3
                'ext_cols_override': selected_features,
            },
            # ---- A2: w/o CA ----
            {
                'name': 'no_ca',
                'desc': 'w/o CA (无通道注意力)',
                'model_class': NoCAModel,
                'model_type': 'vmd',
                'data_mode': 'no_dae',
                'model_kwargs': {'use_dae': True},
                'ext_cols_override': selected_features,
            },
            # ---- A3: w/o TA ----
            {
                'name': 'no_ta',
                'desc': 'w/o TA (无时间注意力)',
                'model_class': NoTAModel,
                'model_type': 'vmd',
                'data_mode': 'no_dae',
                'model_kwargs': {'use_dae': True},
                'ext_cols_override': selected_features,
            },
            # ---- A4: w/o GateFusion ----
            {
                'name': 'no_gate_fusion',
                'desc': 'w/o GateFusion (简单拼接融合)',
                'model_class': NoGateFusionModel,
                'model_type': 'vmd',
                'data_mode': 'no_dae',
                'model_kwargs': {'use_dae': True},
                'ext_cols_override': selected_features,
            },
            # ---- A5: w/o DAE ----
            {
                'name': 'no_dae',
                'desc': 'w/o DAE (无特征编码，用原始5特征)',
                'model_class': VMDCATBiLSTM,
                'model_type': 'vmd',
                'data_mode': 'no_dae',  # 读取原始CSV，5个原始特征
                'model_kwargs': {'use_dae': False},  # 不使用DAE编码
                'ext_cols_override': selected_features,
            },
        ]
    else:
        # ===== 传统预处理DAE模式 =====
        # 所有VMD模型读取_encoded.csv（3个编码特征）
        configs = [
            {
                'name': 'full',
                'desc': 'VMD-CAT-BiLSTM (Full)',
                'model_class': VMDCATBiLSTM,
                'model_type': 'vmd',
                'data_mode': 'vmd_dae',
                'model_kwargs': {},
                'ext_cols_override': None,
            },
            {
                'name': 'no_vmd',
                'desc': 'w/o VMD (无VMD分解)',
                'model_class': NoVMDModel,
                'model_type': 'single',
                'data_mode': 'no_vmd',
                'model_kwargs': {},
                'ext_cols_override': None,
            },
            {
                'name': 'no_ca',
                'desc': 'w/o CA (无通道注意力)',
                'model_class': NoCAModel,
                'model_type': 'vmd',
                'data_mode': 'vmd_dae',
                'model_kwargs': {},
                'ext_cols_override': None,
            },
            {
                'name': 'no_ta',
                'desc': 'w/o TA (无时间注意力)',
                'model_class': NoTAModel,
                'model_type': 'vmd',
                'data_mode': 'vmd_dae',
                'model_kwargs': {},
                'ext_cols_override': None,
            },
            {
                'name': 'no_gate_fusion',
                'desc': 'w/o GateFusion (简单拼接融合)',
                'model_class': NoGateFusionModel,
                'model_type': 'vmd',
                'data_mode': 'vmd_dae',
                'model_kwargs': {},
                'ext_cols_override': None,
            },
            {
                'name': 'no_dae',
                'desc': 'w/o DAE (无特征编码，用原始5特征)',
                'model_class': VMDCATBiLSTM,
                'model_type': 'vmd',
                'data_mode': 'no_dae',
                'model_kwargs': {},
                'ext_cols_override': selected_features,
            },
        ]

    return configs

# ======================== 2. 运行单个消融实验 =======================
def run_single_experiment(config: dict, epochs: int, device: str) -> dict:
    """
    运行单个消融实验并返回评估结果。

    该函数根据配置构建数据加载器和模型，执行训练过程，保存训练记录和模型权重，
    然后在测试集上评估模型性能，最后返回包含关键指标的汇总结果。

    Args:
        config (dict): 实验配置字典，包含以下键：
            - name (str): 实验名称标识
            - desc (str): 实验描述信息
            - model_class (nn.Module): 模型类
            - model_type (str): 模型类型 ('vmd' 或 'single')
            - data_mode (str): 数据模式
            - ext_cols_override (list or None): 外部特征列覆盖设置
        epochs (int): 训练轮数
        device (str): 计算设备 ('cpu' 或 'cuda')

    Returns:
        dict: 实验结果汇总，包含以下键：
            - name (str): 实验名称
            - description (str): 实验描述
            - RMSE (float): 均方根误差
            - MAE (float): 平均绝对误差
            - MAPE (float): 平均绝对百分比误差
            - n_params (int): 可训练参数数量
            - train_time_min (float): 训练耗时（分钟）
            - model_type (str): 模型类型
            - data_mode (str): 数据模式

    Example:
        config = get_ablation_configs()[0]
        result = run_single_experiment(config, epochs=50, device='cuda')
        print(result['RMSE'], result['MAE'])
    """
    name = config['name']
    desc = config['desc']
    model_class = config['model_class']
    model_type = config['model_type']
    data_mode = config['data_mode']
    ext_cols_override = config['ext_cols_override']
    model_kwargs_extra = config.get('model_kwargs', {})

    print("\n" + "=" * 70)
    print(f"🧪 消融实验: {desc} ({name})")
    print(f"   模型类型: {model_type} | 数据模式: {data_mode}")
    if DAE_END_TO_END:
        use_dae = model_kwargs_extra.get('use_dae', False)
        print(f"   DAE端到端: {'✅ 启用' if use_dae else '❌ 禁用'}")
    print("=" * 70)

    # ---- 2.1 构建数据加载器 ----
    train_loader, val_loader, test_loader, n_imf, n_ext = get_dataloaders(
        data_dir=DATA_PROCESSED_DIR,
        target_col=DEMAND_COL,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        data_mode=data_mode,
        ext_cols_override=ext_cols_override,
    )

    # ---- 2.2 构建模型 ----
    common_kwargs = dict(
        tcn_hidden=TCN_HIDDEN,
        tcn_output=TCN_OUTPUT,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        unified_dim=UNIFIED_DIM,
        dropout=DROPOUT,
        share_subnet=SHARE_SUBNET,
    )
    # 合并额外的模型参数（如 use_dae）
    common_kwargs.update(model_kwargs_extra)

    if model_type == 'vmd':
        model = model_class(
            n_imf=n_imf,
            n_external=n_ext,
            **common_kwargs,
        )
    else:
        # single模式不需要n_imf和share_subnet
        single_kwargs = {k: v for k, v in common_kwargs.items() if k != 'share_subnet'}
        model = model_class(
            n_external=n_ext,
            **single_kwargs,
        )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   可训练参数量: {total_params:,}")

    # ---- 2.3 训练 ----
    model_save_path = os.path.join(ABLATION_MODELS_DIR, f'{name}_model.pth')
    record_save_path = os.path.join(ABLATION_RECORDS_DIR, f'{name}_records.csv')

    start_time = time.time()

    training_records = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=LEARNING_RATE,
        device=device,
        weight_decay=WEIGHT_DECAY,
        model_type=model_type,
        save_path=model_save_path,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
    )

    train_time = time.time() - start_time

    # 保存训练记录
    os.makedirs(os.path.dirname(record_save_path), exist_ok=True)
    records_df = pd.DataFrame(training_records)
    records_df.to_csv(record_save_path, index=False)
    print(f"   训练记录已保存: {record_save_path}")
    print(f"   训练耗时: {train_time / 60:.2f} 分钟")

    # ---- 2.4 加载最优模型并在测试集上评测 ----
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    eval_results = evaluate_model(model, test_loader, device, model_type)

    print(f"\n   📊 测试集评估结果:")
    print(f"      RMSE:  {eval_results['RMSE']:.4f}")
    print(f"      MAE:   {eval_results['MAE']:.4f}")
    print(f"      MAPE:  {eval_results['MAPE']:.2f}%")

    # ---- 2.5 汇总结果 ----
    result = {
        'name': name,
        'description': desc,
        'RMSE': eval_results['RMSE'],
        'MAE': eval_results['MAE'],
        'MAPE': eval_results['MAPE'],
        'n_params': eval_results['n_params'],
        'train_time_min': round(train_time / 60, 2),
        'model_type': model_type,
        'data_mode': data_mode,
    }

    return result

# ======================== 3. 主函数 =======================
def main():
    """
    消融实验主函数。

    该函数负责协调整个消融实验流程，包括：
    1. 解析命令行参数（训练轮数、是否跳过Full模型）
    2. 初始化计算设备和输出目录
    3. 获取并过滤实验配置
    4. 依次运行所有消融实验，捕获异常并记录失败信息
    5. 汇总所有实验结果，支持合并之前的Full模型结果
    6. 按固定顺序排序并保存汇总结果到CSV文件

    Returns:
        None: 函数无返回值，结果直接保存到文件

    Example:
    """
    import argparse
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--skip_full', action='store_true', help='跳过Full模型（已有训练结果时使用）')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")
    print(f"   训练轮数: {args.epochs}")
    print(f"   DAE端到端训练: {'✅ 启用' if DAE_END_TO_END else '❌ 禁用'}")
    if DAE_END_TO_END:
        from configs.config import RECON_LOSS_WEIGHT
        print(f"   重构损失权重: {RECON_LOSS_WEIGHT}")

    # 确保输出目录存在
    os.makedirs(ABLATION_MODELS_DIR, exist_ok=True)
    os.makedirs(ABLATION_RECORDS_DIR, exist_ok=True)

    # 获取实验配置
    all_configs = get_ablation_configs()

    # 如果跳过Full模型，移除第一个配置
    if args.skip_full:
        all_configs = [c for c in all_configs if c['name'] != 'full']
        print("⚠️ 跳过 Full 模型训练（使用已有结果）")

    # 运行所有实验
    all_results = []
    for i, config in enumerate(all_configs):
        print(f"\n{'🔹' * 35}")
        print(f" 进度: [{i + 1}/{len(all_configs)}]")
        print(f"{'🔹' * 35}")

        try:
            result = run_single_experiment(config, args.epochs, device)
            all_results.append(result)
        except Exception as e:
            print(f"❌ 实验 {config['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'name': config['name'],
                'description': config['desc'],
                'RMSE': float('nan'),
                'MAE': float('nan'),
                'MAPE': float('nan'),
                'n_params': 0,
                'train_time_min': 0,
                'model_type': config['model_type'],
                'data_mode': config['data_mode'],
                'error': str(e),
            })

    # ---- 汇总结果 ----
    results_df = pd.DataFrame(all_results)

    # 如果跳过了Full模型，尝试加载之前的结果合并
    if args.skip_full and os.path.exists(ABLATION_RESULTS_PATH):
        prev_df = pd.read_csv(ABLATION_RESULTS_PATH)
        full_row = prev_df[prev_df['name'] == 'full']
        if len(full_row) > 0:
            results_df = pd.concat([full_row, results_df], ignore_index=True)
            print("\n✅ 已合并之前 Full 模型的结果")

    # 按固定顺序排列
    order = ['full', 'no_vmd', 'no_ca', 'no_ta', 'no_gate_fusion', 'no_dae']
    results_df['sort_key'] = results_df['name'].apply(lambda x: order.index(x) if x in order else 99)
    results_df = results_df.sort_values('sort_key').drop(columns=['sort_key']).reset_index(drop=True)

    # 保存汇总结果
    results_df.to_csv(ABLATION_RESULTS_PATH, index=False)
    print(f"\n{'=' * 70}")
    print(f"📋 消融实验汇总结果")
    print(f"{'=' * 70}")
    print(results_df[['description', 'RMSE', 'MAE', 'MAPE', 'n_params', 'train_time_min']].to_string(index=False))
    print(f"\n💾 结果已保存至: {ABLATION_RESULTS_PATH}")

# ======================== 测试 =======================
if __name__ == '__main__':
    main()