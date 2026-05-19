"""
文件位置: experiments/ablation/no_vmd.py
作用: 消融实验模型 A1 —— 去除 VMD 分解的对比模型（含容量补偿）
      将原始demand序列与DAE编码后的外部特征直接拼接为单一输入，
      通过 CA-TCN → TA-BiLSTM → GateFusion → PredictionHead 单分支结构进行预测。
      用于验证VMD分解对模型性能的贡献：若该模型指标明显下降，则证明VMD分解有效。

容量补偿说明:
      Full模型通过VMD分解将demand拆为4个IMF，每个IMF独立走一次SubNet（共享参数），
      4次前向传播的预测值累加，等效于4条建模路径，具有更强的多尺度建模能力。
      NoVMD只有1条路径，直接处理未分解的原始信号，容量严重不足。
      为确保消融实验公平性，NoVMD引入capacity_factor（默认2.0）将隐藏维度放大，
      同时增加TCN层数以弥补缺乏VMD带来的多尺度分解能力。

      支持端到端DAE训练：当use_dae=True时，DAE编码器在forward中编码外部特征。
所包含的类:
  - NoVMDModel(nn.Module): 无VMD分解的消融模型（含容量补偿）
是否需要创建: 是，本文件需要创建（P3 消融实验专用）
相关文件更新:
  - 依赖 src/models/ca_tcn.py (CATCN)
  - 依赖 src/models/ta_bilstm.py (TABiLSTM)
  - 依赖 src/models/fusion.py (GateFusion, PredictionHead)
  - 输入数据由 data_mode='no_vmd' 的 DataLoader 提供，形状为 (batch, seq_len, 1+n_external)
  - 输出预测值与注意力权重，供 P3 实验评估使用
"""
# ======================== 0. 引入包 =======================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ca_tcn import CATCN
from src.models.ta_bilstm import TABiLSTM
from src.models.fusion import GateFusion, PredictionHead
from src.features.dae_encoder import DenoisingAE
from configs.config import (
    DAE_HIDDEN_DIM,             # DAE 隐藏层维度
    NOISE_STD,                  # DAE 噪声标准差
    DAE_MODEL_FILE_PATH,        # DAE 模型文件路径
    DROPOUT,                    # dropout 概率
)

# ======================== 1. 去除vmd的模型 =======================
class NoVMDModel(nn.Module):
    """
    A1: w/o VMD 消融模型

    去掉VMD变分模态分解，将原始demand序列与外部特征合并为单一输入，
    通过单个 CA-TCN → TA-BiLSTM → GateFusion → PredictionHead 分支进行预测。
    与Full模型的区别：Full模型有4个并行SubNet（每个处理一个IMF分量），
    而本模型只有1个分支，直接处理未分解的原始信号。

    Attributes:
        input_dim (int): 输入维度 = 1(demand) + n_external
        ca_tcn (CATCN): 通道注意力时间卷积网络
        local_pool (nn.AdaptiveAvgPool1d): 局部特征全局池化
        ta_bilstm (TABiLSTM): 时间注意力双向LSTM
        fusion (GateFusion): 门控融合模块
        prediction_head (PredictionHead): 预测头
    """

    def __init__(
            self,
            n_external: int,
            tcn_hidden: int = 64,
            tcn_output: int = 32,
            lstm_hidden: int = 64,
            lstm_layers: int = 2,
            unified_dim: int = 128,
            dropout: float = 0.2,
            use_dae: bool = False,
            dae_pretrained_path: str = None,
            capacity_factor: float = 2.0,
    ):
        """
        初始化 NoVMD 模型（含容量补偿）

        capacity_factor补偿原理：
        Full模型将demand拆为n_imf个IMF，每个IMF独立通过SubNet（共享参数），
        等效n_imf条建模路径的预测累加。NoVMD仅有1条路径，直接建模全频段信号。
        为保证消融公平性，通过capacity_factor放大隐藏维度：
        - tcn_hidden_comp = tcn_hidden * capacity_factor
        - lstm_hidden_comp = lstm_hidden * capacity_factor
        - unified_dim_comp = unified_dim * capacity_factor
        同时增加TCN层数（+2层），弥补VMD缺失导致的多尺度分解能力不足。

        Args:
            n_external (int): 外部特征维度
            tcn_hidden (int): TCN隐藏层维度（补偿前基准值）
            tcn_output (int): TCN输出维度
            lstm_hidden (int): BiLSTM隐藏层维度（补偿前基准值）
            lstm_layers (int): BiLSTM层数
            unified_dim (int): 门控融统一维度（补偿前基准值）
            dropout (float): Dropout比率
            use_dae (bool): 是否使用端到端DAE编码器
            dae_pretrained_path (str): DAE 预训练权重路径
            capacity_factor (float): 容量补偿因子，默认2.0（将隐藏维度翻倍）
        """
        super(NoVMDModel, self).__init__()

        self.use_dae = use_dae

        # ---- 容量补偿：放大隐藏维度以弥补单分支劣势 ----
        tcn_hidden_comp = int(tcn_hidden * capacity_factor)
        lstm_hidden_comp = int(lstm_hidden * capacity_factor)
        unified_dim_comp = int(unified_dim * capacity_factor)
        # TCN层数增加2层以弥补VMD多尺度分解能力的缺失
        tcn_num_layers_comp = 4 + 2  # 默认4层 → 6层

        # ---- DAE编码器（端到端训练模式） ----
        if use_dae:
            dae_output_dim = DAE_HIDDEN_DIM[-1]
            self.dae = DenoisingAE(
                input_dim=n_external,
                hidden_dim=DAE_HIDDEN_DIM,
                noise_std=NOISE_STD,
                dropout=dropout,
            )
            pretrained = dae_pretrained_path or DAE_MODEL_FILE_PATH
            if os.path.exists(pretrained):
                self.dae.load_state_dict(torch.load(pretrained, map_location='cpu'))
            self.dae_output_dim = dae_output_dim
            input_dim = 1 + dae_output_dim
        else:
            input_dim = 1 + n_external

        self.ca_tcn = CATCN(
            input_dim=input_dim,
            hidden_dim=tcn_hidden_comp,
            output_dim=tcn_output,
            num_layers=tcn_num_layers_comp,
            dropout=dropout,
        )

        self.local_pool = nn.AdaptiveAvgPool1d(1)

        self.ta_bilstm = TABiLSTM(
            input_dim=tcn_output,
            hidden_dim=lstm_hidden_comp,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        self.fusion = GateFusion(
            local_dim=tcn_output,
            global_dim=lstm_hidden_comp * 2,
            unified_dim=unified_dim_comp,
        )

        self.prediction_head = PredictionHead(
            input_dim=unified_dim_comp,
            hidden_dims=[unified_dim_comp, unified_dim_comp // 2],
            dropout=dropout,
        )

    def compute_dae_loss(self, raw_externals: torch.Tensor) -> torch.Tensor:
        """
        计算DAE（Denoising Autoencoder）的重构损失。

        该方法用于在端到端训练过程中对DAE编码器进行正则化约束，通过计算重构误差来保证编码质量。
        如果未启用DAE（use_dae=False），则返回零张量以保持损失计算的兼容性。

        优先使用 forward() 中缓存的解码结果，避免重复前向传播导致噪声不一致。
        使用 _cache_valid 标志位判断缓存是否来自当前 forward 调用，
        避免依赖 Python 对象同一性（is 检查），因为 PyTorch 切片操作每次创建新视图对象，
        导致 is 检查在 single 模型下始终为 False。

        Args:
            raw_externals (torch.Tensor): 原始外部特征张量，形状为 (batch_size, seq_length, n_external)

        Returns:
            torch.Tensor: DAE重构损失标量值。如果未启用DAE则返回0.0，否则返回均方误差损失

        Note:
            - 使用self.training标志控制是否在推理时添加噪声
            - 损失函数采用均方误差（MSE）衡量重构质量
            - 输入张量会被展平为二维以匹配DAE的输入格式
        """
        if not self.use_dae:
            return torch.tensor(0.0, device=raw_externals.device)
        # 优先使用 forward() 中缓存的解码结果，避免重复前向传播
        # 注意：不能用 is 检查对象同一性，因为 trainer 中 x[:, :, 1:] 与 forward 中
        # x[:, :, 1:] 是不同的 PyTorch 视图对象（即使底层存储相同）
        if getattr(self, '_cache_valid', False):
            # 验证缓存形状与当前输入一致
            if (self._cached_decoded.shape == raw_externals.shape):
                self._cache_valid = False  # 消费缓存，防止重复使用
                return F.mse_loss(
                    self._cached_decoded.reshape(-1, self._cached_decoded.shape[-1]),
                    self._cached_raw.reshape(-1, self._cached_raw.shape[-1])
                )
        # 降级：如果缓存不可用，则重新计算
        batch_size, seq_len, n_raw = raw_externals.shape
        ext_flat = raw_externals.reshape(-1, n_raw)
        _, decoded_flat = self.dae(ext_flat, noise=self.training)
        return F.mse_loss(decoded_flat, ext_flat)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, 1 + n_external)
                             包含原始demand和外部特征

        Returns:
            tuple:
                - y_pred (torch.Tensor): 预测值，形状为 (batch_size,)
                - atten_weights (torch.Tensor): TA-BiLSTM的注意力权重
        """
        # DAE编码（端到端模式）
        if self.use_dae:
            demand = x[:, :, :1]  # (batch, seq_len, 1)
            raw_features = x[:, :, 1:]  # (batch, seq_len, n_external)
            batch_size, seq_len, n_raw = raw_features.shape

            # 编码外部特征
            ext_flat = raw_features.reshape(-1, n_raw)
            encoded_flat, decoded_flat = self.dae(ext_flat, noise=self.training)
            encoded = encoded_flat.reshape(batch_size, seq_len, -1)

            # 缓存解码结果，供 compute_dae_loss() 使用，避免重复前向传播
            self._cached_decoded = decoded_flat.reshape(batch_size, seq_len, n_raw)
            self._cached_raw = raw_features
            self._cache_valid = True  # 标记缓存有效（当前 forward 调用产生）

            # 重新拼接: demand + encoded_features
            x = torch.cat([demand, encoded], dim=-1)

        f_local_seq = self.ca_tcn(x)
        f_local = self.local_pool(f_local_seq.transpose(1, 2)).squeeze(-1)
        f_global, atten_weights = self.ta_bilstm(f_local_seq)
        f_fused = self.fusion(f_local, f_global)
        y_pred = self.prediction_head(f_fused)
        return y_pred, atten_weights