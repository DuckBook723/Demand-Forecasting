"""
文件位置: experiments/ablation/no_dae.py
消融实验 A5: w/o DAE
      去掉DAE(去噪自编码器)特征编码，直接使用原始5个选中特征代替3个DAE编码特征。
      模型架构与Full模型完全相同，仅 n_external 从 3 变为 5。
      验证DAE特征降维与去噪对预测精度的贡献。

注意: 本模块不定义新的模型类，而是复用 VMDCATBiLSTM，只是传入不同的 n_external 参数。
      关键差异在数据层面：使用 train.csv (含原始5特征) 而非 train_encoded.csv (含3个ENC特征)。
"""

# ======================== 0. 引入包 =======================
from src.models.vmd_cat_bilstm import VMDCATBiLSTM

# A5消融直接复用 VMDCATBiLSTM，仅将 n_external=5（而非3）
# 具体调用方式:
#   model = VMDCATBiLSTM(n_imf=n_imf, n_external=5, ...)
#   dataloaders 使用 data_mode='no_dae'

__all__ = ['VMDCATBiLSTM']