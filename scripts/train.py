"""
文件位置: scripts/train.py
作用: 模型训练主入口
      1. 检查数据流水线是否就绪（train/val/test_encoded.csv 是否存在）
      2. 解析超参数（支持命令行覆盖默认值）
      3. 加载 DataLoader（自动识别 IMF 和外部特征数量）
      4. 构建 VMD-CAT-BiLSTM 模型并启动训练
      5. 训练完成后保存最优模型至指定路径
所包含的函数:
  - check_dataset_ready(): 校验数据目录下的训练/验证/测试文件完整性
  - parse_args(): 命令行参数解析，支持从配置文件或命令行覆盖关键超参数
  - main(): 训练主流程，串联数据检查、DataLoader 构建、模型初始化与训练
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 依赖 configs/config.py 中的路径与超参数配置
  - 依赖 src/data/dataset.py 中的 get_dataloaders
  - 依赖 src/models/vmd_cat_bilstm.py 中的 VMDCATBiLSTM
  - 依赖 src/models/trainer.py 中的 train_model
  - 输出 checkpoints/ 下的模型权重文件 (由 train_model 内部调用)
"""

# ======================== 0. 引入包 =======================
import sys
import os
import argparse
import torch
import time
import pandas as pd
from configs.config import (
    DATA_PROCESSED_DIR,     # 数据处理结果目录
    DEMAND_COL,             # 目标列名
    SEQ_LEN,                # 滑动窗口长度
    EPOCHS,                 # 训练轮次
    BATCH_SIZE,             # 批次大小
    LEARNING_RATE,          # 学习率
    TCN_HIDDEN,             # TCN隐藏层维度
    TCN_OUTPUT,             # TCN输出维度
    LSTM_HIDDEN,            # LSTM隐藏层数
    LSTM_LAYERS,            # LSTM层数
    UNIFIED_DIM,            # 门控融合同一纬度
    DROPOUT,                # dropout比例
    EARLY_STOPPING,         # 早停开关
    PATIENCE,               # 早停耐心值
    MODEL_FILE_PATH,        # 模型保存路径
    TRAIN_RECORD_FILE_PATH, # 训练记录保存路径
    WEIGHT_DECAY,           # 权重衰减系数
    SHARE_SUBNET,           # SubNet参数共享
    DAE_END_TO_END,         # DAE端到端训练开关
)
from src.data.dataset import get_dataloaders
from src.models.vmd_cat_bilstm import VMDCATBiLSTM
from src.models.trainer import train_model

# ======================== 1. 判断数据集是否准备就绪 =======================
def check_dataset_ready():
    """
    检查训练所需的数据文件是否已准备就绪

    验证数据预处理流水线的输出文件是否存在，包括训练集、验证集和测试集的编码后CSV文件。
    如果缺少任何必需文件，打印详细的错误信息和数据预处理步骤指南，然后退出程序。

    Required Files:
        - train_encoded.csv: 训练集数据
        - val_encoded.csv: 验证集数据
        - test_encoded.csv: 测试集数据

    Note:
        如果文件缺失，会提示用户按顺序运行以下脚本：
        1. generate_data.py - 生成原始仿真数据
        2. preprocess.py - VMD分解、归一化、数据划分
        3. feature_selector.py - 特征筛选
        4. dae_encoder.py - DAE特征降维
    """
    if DAE_END_TO_END:
        required_files = [os.path.join(DATA_PROCESSED_DIR, f)
                          for f in ['train.csv', 'val.csv', 'test.csv']]
    else:
        required_files = [os.path.join(DATA_PROCESSED_DIR, f)
                          for f in ['train_encoded.csv', 'val_encoded.csv', 'test_encoded.csv']]

    for f in required_files:
        if not os.path.exists(f):
            print("\n" + "=" * 60)
            print("❌ 错误：未找到训练所需的数据文件！")
            print(f"   缺失: {f}")
            print("\n💡 请按以下顺序依次运行数据预处理流水线：")
            print("   1. python src/data/generate_data.py      (生成原始仿真数据)")
            print("   2. python src/data/preprocess.py         (VMD分解、归一化、划分)")
            print("   3. python src/features/feature_selector.py (特征筛选)")
            print("   4. python src/features/dae_encoder.py    (DAE特征降维)")
            print("=" * 60 + "\n")
            sys.exit(1)

# ======================== 2. 参数解析 =======================
def parse_args():
    """
    命令行参数解析函数。

    解析训练过程中所需的超参数配置，支持从命令行覆盖默认值。
    所有参数的默认值均从 configs.config 模块导入的全局常量中获取。

    Returns:
        argparse.Namespace: 包含以下属性的命名空间对象：
            - seq_len (int): 输入时间窗口长度（回看过去几个月）
            - batch_size (int): Batch大小
            - epochs (int): 最大训练轮数
            - lr (float): 初始学习率
            - tcn_hidden (int): TCN隐藏层维度
            - tcn_output (int): TCN输出维度 / BiLSTM输入维度
            - lstm_hidden (int): BiLSTM隐藏层维度
            - lstm_layers (int): BiLSTM层数
            - unified_dim (int): 门控融合统一维度
            - weight_decay (float): 权重衰减系数，用于L2正则化
            - dropout (float): Dropout比率，用于防止过拟合
            - early_stopping (bool): 是否启用早停机制
            - patience (int): 早停耐心值，即验证集性能不提升时的最大等待轮数
            - share_subnet (bool): 是否共享SubNet参数（所有IMF分量共用同一子网络）

    Example:
        >>> args = parse_args()
        >>> print(args.lr)  # 默认值来自 configs.config.LEARNING_RATE
        >>> # 可通过命令行覆盖: python train.py --lr 0.001 --epochs 100
    """
    parser = argparse.ArgumentParser(description="Train VMD-CAT-BiLSTM for Grid Material Demand")
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN, help='输入时间窗口长度 (回看过去几个月)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch大小')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='初始学习率')
    parser.add_argument('--tcn_hidden', type=int, default=TCN_HIDDEN, help='TCN隐藏层维度')
    parser.add_argument('--tcn_output', type=int, default=TCN_OUTPUT, help='TCN输出维度 / BiLSTM输入维度')
    parser.add_argument('--lstm_hidden', type=int, default=LSTM_HIDDEN, help='BiLSTM隐藏层维度')
    parser.add_argument('--lstm_layers', type=int, default=LSTM_LAYERS, help='BiLSTM层数')
    parser.add_argument('--unified_dim', type=int, default=UNIFIED_DIM, help='门控融合统一维度')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='权重衰减系数')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout比率')
    parser.add_argument('--early_stopping', type=bool, default=EARLY_STOPPING, help='是否启用早停')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='早停耐心值')
    parser.add_argument('--share_subnet', type=bool, default=SHARE_SUBNET, help='是否共享SubNet参数')
    return parser.parse_args()

# ======================== 3. 训练主函数 =======================
def main():
    """
    训练流程的主入口函数

    完整执行以下步骤：
    1. 解析命令行参数获取超参数配置
    2. 检测设备可用性（优先使用GPU）
    3. 验证数据文件是否就绪
    4. 加载数据并构建DataLoader
    5. 初始化VMDCATBiLSTM模型并输出模型信息
    6. 执行训练循环（包含验证和最优模型保存）
    7. 输出训练完成信息和总耗时

    训练过程中会自动保存验证损失最优的模型到MODEL_FILE_PATH指定位置。

    Raises:
        SystemExit: 当训练集或验证集样本数为0时退出程序

    Note:
        - 自动检测CUDA设备，不可用时降级到CPU
        - 训练前验证数据集非空，避免无效训练
        - 显示模型参数量和架构信息便于调试
        - 记录训练总耗时（以分钟为单位）
    """
    parser = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    check_dataset_ready()

    # 根据DAE端到端模式选择数据模式
    if DAE_END_TO_END:
        # 端到端模式：读取原始CSV，模型内部DAE编码
        import json
        from configs.config import FEATURE_SELECTION_FILE_PATH
        selected_features = None
        if os.path.exists(FEATURE_SELECTION_FILE_PATH):
            with open(FEATURE_SELECTION_FILE_PATH, 'r') as f:
                selected_features = json.load(f)
        data_mode = 'no_dae'
        ext_cols_override = selected_features
    else:
        data_mode = 'vmd_dae'
        ext_cols_override = None

    print(f"\n📂 Loading data and building DataLoaders... (data_mode={data_mode})")
    train_loader, val_loader, test_loader, n_imf, n_ext = get_dataloaders(
        data_dir=DATA_PROCESSED_DIR,
        target_col=DEMAND_COL,
        seq_len=parser.seq_len,
        batch_size=parser.batch_size,
        data_mode=data_mode,
        ext_cols_override=ext_cols_override,
    )

    # 验证数据集有效性，确保有足够样本进行训练
    if len(train_loader) == 0 or len(val_loader) == 0:
        print("❌ 错误：训练集或验证集样本数为0！")
        print("💡 请检查 generate_data.py 中的时间跨度是否足够长 (建议至少15年以上)。")
        sys.exit(1)

    print("\n🧠 Initializing VMDCATBiLSTM model...")
    model = VMDCATBiLSTM(
        n_imf=n_imf,
        n_external=n_ext,
        tcn_hidden=parser.tcn_hidden,
        tcn_output=parser.tcn_output,
        lstm_hidden=parser.lstm_hidden,
        lstm_layers=parser.lstm_layers,
        unified_dim=parser.unified_dim,
        dropout=parser.dropout,
        share_subnet=parser.share_subnet,
        use_dae=DAE_END_TO_END,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model Structure: VMDCATBiLSTM")
    print(f"   N_IMF: {n_imf} | N_External: {n_ext}")
    print(f"   Total Trainable Params: {total_params:,}")

    print(f"\n🏋️ Start training (Max Epochs: {parser.epochs}")
    start_time = time.time()

    training_records = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=parser.epochs,
        lr=parser.lr,
        device=device,
        early_stopping=parser.early_stopping,
        patience=parser.patience,
        weight_decay=parser.weight_decay,
        model_type='vmd',
    )

    # 保存训练记录至csv
    if training_records:
        records_df = pd.DataFrame(training_records)
        os.makedirs(os.path.dirname(TRAIN_RECORD_FILE_PATH), exist_ok=True)
        records_df.to_csv(TRAIN_RECORD_FILE_PATH, index=False)
        print(f"💾 Training records saved at: {TRAIN_RECORD_FILE_PATH}")

    end_time = time.time()
    print(f"\n🏁 Training finished! Total time: {(end_time - start_time) / 60:.2f} mins")
    print(f"💾 Best model saved at: {MODEL_FILE_PATH}")

# ======================== 测试 =======================
if __name__ == "__main__":
    main()