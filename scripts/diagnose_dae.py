"""
文件位置: scripts/diagnose_dae.py
作用: 一键诊断DAE编码是否在VMD-CAT-BiLSTM中正确执行

运行方式:
  python scripts/diagnose_dae.py

诊断内容:
  1. 创建VMDCATBiLSTM模型(use_dae=True)
  2. 用虚拟数据测试forward
  3. 验证DAE编码输出维度
  4. 如果出错，给出修复建议
"""

import os
import sys
import torch

# 确保项目根目录在 sys.path 中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    DAE_HIDDEN_DIM, NOISE_STD, DAE_MODEL_FILE_PATH, DROPOUT,
    TCN_HIDDEN, TCN_OUTPUT, LSTM_HIDDEN, LSTM_LAYERS, UNIFIED_DIM,
    VMD_K, TOP_N, FEATURE_SELECTION_FILE_PATH,
)
from src.models.vmd_cat_bilstm import VMDCATBiLSTM
from src.features.dae_encoder import DenoisingAE


def main():
    print("=" * 70)
    print("🔍 DAE编码诊断工具")
    print("=" * 70)

    # 1. 读取特征数
    import json
    n_external = TOP_N
    if os.path.exists(FEATURE_SELECTION_FILE_PATH):
        with open(FEATURE_SELECTION_FILE_PATH, 'r') as f:
            selected_features = json.load(f)
        n_external = len(selected_features)

    print(f"\n📋 配置信息:")
    print(f"   n_external (原始特征数) = {n_external}")
    print(f"   DAE_HIDDEN_DIM = {DAE_HIDDEN_DIM}")
    print(f"   DAE编码输出维度 = {DAE_HIDDEN_DIM[-1]}")
    print(f"   VMD_K (IMF数) = {VMD_K}")
    print(f"   DAE权重文件 = {DAE_MODEL_FILE_PATH}")

    # 2. 独立测试DAE
    print(f"\n{'=' * 70}")
    print(f"🧪 测试1: 独立DAE编码器")
    print(f"{'=' * 70}")

    dae = DenoisingAE(
        input_dim=n_external,
        hidden_dim=DAE_HIDDEN_DIM,
        noise_std=NOISE_STD,
        dropout=DROPOUT,
    )

    # 加载预训练权重
    if os.path.exists(DAE_MODEL_FILE_PATH):
        state_dict = torch.load(DAE_MODEL_FILE_PATH, map_location='cpu')

        # 检查维度匹配
        current_state = dae.state_dict()
        mismatch = False
        for key, param in current_state.items():
            if key in state_dict:
                if state_dict[key].shape != param.shape:
                    print(f"  ❌ 维度不匹配: {key} 期望 {param.shape} 文件 {state_dict[key].shape}")
                    mismatch = True
            else:
                print(f"  ❌ 缺失key: {key}")
                mismatch = True

        if mismatch:
            print(f"\n  ❌ DAE预训练权重与当前架构不兼容!")
            print(f"  💡 请重新训练DAE: python src/features/dae_encoder.py")
            return
        else:
            dae.load_state_dict(state_dict, strict=True)
            print(f"  ✅ DAE权重加载成功")
    else:
        print(f"  ⚠️ DAE权重文件不存在，使用随机初始化")

    # 测试DAE编码
    dae.eval()
    with torch.no_grad():
        test_input = torch.randn(4, n_external)  # 4个样本, n_external维
        encoded, decoded = dae(test_input, noise=False)
        print(f"  输入: {test_input.shape}")
        print(f"  编码: {encoded.shape} (期望 [4, {DAE_HIDDEN_DIM[-1]}])")
        print(f"  解码: {decoded.shape} (期望 [4, {n_external}])")

        if encoded.shape[-1] == DAE_HIDDEN_DIM[-1]:
            print(f"  ✅ DAE编码输出维度正确!")
        else:
            print(f"  ❌ DAE编码输出维度错误!")
            return

    # 3. 测试VMDCATBiLSTM的forward
    print(f"\n{'=' * 70}")
    print(f"🧪 测试2: VMDCATBiLSTM完整forward (use_dae=True)")
    print(f"{'=' * 70}")

    n_imf = VMD_K
    model = VMDCATBiLSTM(
        n_imf=n_imf,
        n_external=n_external,
        tcn_hidden=TCN_HIDDEN,
        tcn_output=TCN_OUTPUT,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        unified_dim=UNIFIED_DIM,
        dropout=DROPOUT,
        share_subnet=True,
        use_dae=True,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数量: {total_params:,}")

    # 模拟数据
    batch_size = 4
    seq_len = 12
    imfs = torch.randn(batch_size, seq_len, n_imf)
    externals = torch.randn(batch_size, seq_len, n_external)

    print(f"  输入: imfs={imfs.shape}, externals={externals.shape}")

    try:
        model.eval()
        with torch.no_grad():
            pred, weights = model(imfs, externals)
        print(f"  输出: pred={pred.shape}, weights={len(weights)}个")
        print(f"\n  ✅✅✅ VMDCATBiLSTM forward 成功! DAE编码正常工作!")
    except RuntimeError as e:
        print(f"\n  ❌ VMDCATBiLSTM forward 失败!")
        print(f"  错误: {e}")
        print(f"\n  🔧 诊断建议:")
        print(f"  1. 删除所有 __pycache__ 目录:")
        print(
            f"     在项目根目录运行: python -c \"import shutil; [shutil.rmtree(d) for d in __import__('pathlib').Path('.').rglob('__pycache__')]\"")
        print(f"  2. 重新训练DAE: python src/features/dae_encoder.py")
        print(f"  3. 检查 src/models/vmd_cat_bilstm.py 是否为最新版本")

    # 4. 测试w/o DAE模式
    print(f"\n{'=' * 70}")
    print(f"🧪 测试3: VMDCATBiLSTM forward (use_dae=False, 对比)")
    print(f"{'=' * 70}")

    model_no_dae = VMDCATBiLSTM(
        n_imf=n_imf,
        n_external=n_external,
        tcn_hidden=TCN_HIDDEN,
        tcn_output=TCN_OUTPUT,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        unified_dim=UNIFIED_DIM,
        dropout=DROPOUT,
        share_subnet=True,
        use_dae=False,
    )

    try:
        model_no_dae.eval()
        with torch.no_grad():
            pred2, weights2 = model_no_dae(imfs, externals)
        print(f"  输出: pred={pred2.shape}")
        print(f"  ✅ w/o DAE模式正常!")
    except RuntimeError as e:
        print(f"  ❌ w/o DAE模式也失败: {e}")

    print(f"\n{'=' * 70}")
    print(f"🏁 诊断完成")


if __name__ == '__main__':
    main()