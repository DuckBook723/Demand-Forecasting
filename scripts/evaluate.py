# scripts/evaluate.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from configs.config import MODEL_FILE_PATH, DATA_PROCESSED_DIR, FIGURES_DIR, DEMAND_COL,DEMAND_MIN_MAX_SCALER_PATH
from src.data.dataset import get_dataloaders
from src.models.vmd_cat_bilstm import VMDCATBiLSTM
from src.models.trainer import calculate_metrics


def evaluate_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据与模型参数
    _, _, test_loader, n_imf, n_external = get_dataloaders(DATA_PROCESSED_DIR, DEMAND_COL, seq_len=12, batch_size=32)

    model = VMDCATBiLSTM(
        n_imf=n_imf, n_external=n_external,
        tcn_hidden=64, tcn_output=32, lstm_hidden=64, unified_dim=128
    ).to(device)

    # 加载训练好的权重
    if not os.path.exists(MODEL_FILE_PATH):
        print("❌ 未找到训练好的模型文件！")
        return
    model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=device))
    model.eval()
    print("✅ 模型权重加载成功！")

    # 2. 加载归一化参数用于反归一化
    try:
        minmax_dict = joblib.load(DEMAND_MIN_MAX_SCALER_PATH)
        d_min, d_max = minmax_dict['min'], minmax_dict['max']
    except:
        print("⚠️ 未找到反归一化参数，将使用归一化值绘图！")
        d_min, d_max = 0, 1

    # 3. 在测试集上推理
    all_y_true_norm = []
    all_y_pred_norm = []

    with torch.no_grad():
        for imfs, exts, y in test_loader:
            imfs, exts = imfs.to(device), exts.to(device)
            y_pred, _ = model(imfs, exts)

            all_y_true_norm.extend(y.cpu().numpy().flatten().tolist())
            all_y_pred_norm.extend(y_pred.cpu().numpy().flatten().tolist())

    all_y_true_norm = np.array(all_y_true_norm)
    all_y_pred_norm = np.array(all_y_pred_norm)

    # 4. 反归一化 (还原成真实的物资吨数)
    y_true_real = all_y_true_norm * (d_max - d_min) + d_min
    y_pred_real = all_y_pred_norm * (d_max - d_min) + d_min
    y_pred_real = np.maximum(y_pred_real, 0)  # 需求不能为负

    # 5. 计算真实业务指标
    metrics = calculate_metrics(y_true_real, y_pred_real, is_normalized=False)
    print("\n" + "=" * 50)
    print("📊 测试集最终评估结果 (反归一化后真实数值)")
    print(f"   RMSE: {metrics['RMSE']:.2f}")
    print(f"   MAE:  {metrics['MAE']:.2f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print("=" * 50 + "\n")

    # 6. 绘制对比图
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_real, label='Actual Demand', color='#2E86AB', linewidth=2, marker='o', markersize=4)
    plt.plot(y_pred_real, label='Predicted Demand', color='#A23B72', linewidth=2, marker='s', markersize=4,
             linestyle='--')

    plt.title('Grid Material Demand Prediction (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step (Month)')
    plt.ylabel('Demand')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(FIGURES_DIR, 'test_prediction_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 预测结果对比图已保存至: {save_path}")
    plt.close()


if __name__ == '__main__':
    evaluate_and_plot()