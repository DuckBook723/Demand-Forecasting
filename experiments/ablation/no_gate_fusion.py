"""
文件位置: experiments/ablation/no_gate_fusion.py
作用: 消融实验模型 A? —— 去除门控融合机制(GateFusion)的对比模型
      完整保留了 VMD-CAT-BiLSTM 的 VMD 分解、CA-TCN 局部特征提取、
      TA-BiLSTM 全局时序建模和预测头架构，但将 GateFusion 替换为简单的
      ConcatFusion（特征拼接 + 线性映射），去除了可学习的门控权重。
      用于验证自适应门控融合相对于简单拼接的优越性：
      若该模型指标明显差于完整模型，则证明门控融合有效。
      支持端到端DAE训练：当use_dae=True时，DAE编码器在forward中编码原始特征。
所包含的类:
  - ConcatFusion(nn.Module): 拼接融合模块，直接拼接局部/全局特征并通过线性层映射
  - SubNetNoGF(nn.Module): 使用 ConcatFusion 的单分量预测子网络
  - NoGateFusionModel(nn.Module): 无门控融合的完整消融模型
是否需要创建: 是，本文件需要创建（P3 消融实验专用）
相关文件更新:
  - 依赖 src/models/ca_tcn.py (CATCN)
  - 依赖 src/models/ta_bilstm.py (TABiLSTM)
  - 依赖 src/models/fusion.py (仅引用 PredictionHead，GateFusion 被替换)
  - 输入数据与完整模型相同：IMF 分量 + DAE 编码外部特征
  - 输出预测值与注意力权重，供 P3 实验评估使用
"""

# ======================== 0. 引入包 =======================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ca_tcn import CATCN
from src.models.ta_bilstm import TABiLSTM
from src.models.fusion import PredictionHead
from src.features.dae_encoder import DenoisingAE
from configs.config import (
    DAE_HIDDEN_DIM,             # DAE 隐藏层维度
    NOISE_STD,                  # DAE 噪声标准差
    DAE_MODEL_FILE_PATH,        # DAE 模型文件路径
    DROPOUT,                    # dropout 概率
)

# ======================== 1. 拼接融合模块 =======================
class ConcatFusion(nn.Module):
    """
    简单拼接融合模块，不含门控机制。

    该模块通过独立的线性投影将局部特征和全局特征映射到统一维度，
    然后直接拼接并通过线性层进行融合。这是消融实验中用于验证
    门控融合机制作用的简化版本。

    Args:
        local_dim (int): 局部特征的输入维度
        global_dim (int): 全局特征的输入维度
        unified_dim (int, optional): 统一映射维度，默认为64

    Example:
        >>> fusion = ConcatFusion(local_dim=32, global_dim=128, unified_dim=64)
        >>> local_feat = torch.randn(32, 32)  # batch_size=32
        >>> global_feat = torch.randn(32, 128)
        >>> fused = fusion(local_feat, global_feat)  # output shape: (32, 64)
    """

    def __init__(self, local_dim: int, global_dim: int, unified_dim: int = 64):
        super(ConcatFusion, self).__init__()

        # 局部特征和全局特征的独立投影层
        self.local_proj = nn.Linear(local_dim, unified_dim)
        self.global_proj = nn.Linear(global_dim, unified_dim)

        # 直接拼接 + 线性映射，无门控
        self.fusion_proj = nn.Linear(unified_dim * 2, unified_dim)

    def forward(self, local_feature: torch.Tensor, global_feature: torch.Tensor):
        """
        前向传播函数。

        Args:
            local_feature (torch.Tensor): 局部特征张量，形状为 (batch_size, local_dim) 或 (batch_size, seq_len, local_dim)
            global_feature (torch.Tensor): 全局特征张量，形状为 (batch_size, global_dim) 或 (batch_size, seq_len, global_dim)

        Returns:
            torch.Tensor: 融合后的特征张量，形状为 (batch_size, unified_dim)
        """
        # 如果输入是3D张量，则沿序列维度取平均
        if local_feature.dim() == 3:
            local_feature = local_feature.mean(dim=1)
        if global_feature.dim() == 3:
            global_feature = global_feature.mean(dim=1)

        # 独立投影并应用ReLU激活
        local_unified = torch.relu(self.local_proj(local_feature))
        global_unified = torch.relu(self.global_proj(global_feature))

        # 直接拼接后通过线性层
        concat = torch.cat([local_unified, global_unified], dim=1)
        f_fused = self.fusion_proj(concat)

        return f_fused

# ======================== 2. 使用拼接融合的SubNet =======================
class SubNetNoGF(nn.Module):
    """
    不含门控融合（Gate Fusion）的子网络模块。

    该网络结合了带通道注意力的TCN和带时间注意力的BiLSTM，通过简单拼接融合机制整合局部特征和全局特征，
    最后通过预测头输出预测结果。这是消融实验中用于验证门控融合机制作用的变体版本。

    Args:
        input_dim (int): 输入特征维度
        tcn_hidden (int, optional): TCN隐藏层维度，默认为64
        tcn_output (int, optional): TCN输出维度，默认为32
        lstm_hidden (int, optional): LSTM隐藏层维度，默认为64
        lstm_layers (int, optional): LSTM层数，默认为2
        unified_dim (int, optional): 融合后的统一维度，默认为128

    Example:
        >>> model = SubNetNoGF(input_dim=10, tcn_hidden=64, tcn_output=32,
        ...                    lstm_hidden=64, lstm_layers=2, unified_dim=128)
        >>> x = torch.randn(32, 100, 10)  # batch_size=32, seq_len=100
        >>> y_pred, atten_weights = model(x)
    """

    def __init__(self, input_dim: int, tcn_hidden: int = 64, tcn_output: int = 32,
                 lstm_hidden: int = 64, lstm_layers: int = 2, unified_dim: int = 128, dropout: float = 0.2):
        super(SubNetNoGF, self).__init__()

        # 使用带通道注意力的TCN提取局部时序特征
        self.ca_tcn = CATCN(
            input_dim=input_dim,
            hidden_dim=tcn_hidden,
            output_dim=tcn_output,
            dropout=dropout,
        )

        # 全局池化获取序列级特征
        self.local_pool = nn.AdaptiveAvgPool1d(1)

        # 带时间注意力的BiLSTM提取全局时序特征
        self.ta_bilstm = TABiLSTM(
            input_dim=tcn_output,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        # 使用ConcatFusion替代GateFusion
        self.fusion = ConcatFusion(
            local_dim=tcn_output,
            global_dim=lstm_hidden * 2,
            unified_dim=unified_dim,
        )

        # 预测头生成最终预测结果
        self.prediction_head = PredictionHead(
            input_dim=unified_dim,
            hidden_dims=[64, 32],
            dropout=dropout,
        )

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, input_dim)

        Returns:
            tuple: (预测值, 注意力权重)
                - y_pred (torch.Tensor): 预测结果，形状为 (batch_size,)
                - atten_weights (torch.Tensor): 时间注意力权重
        """
        f_local_seq = self.ca_tcn(x)
        f_local = self.local_pool(f_local_seq.transpose(1, 2)).squeeze(-1)
        f_global, atten_weights = self.ta_bilstm(f_local_seq)
        f_fused = self.fusion(f_local, f_global)
        y_pred = self.prediction_head(f_fused)
        return y_pred, atten_weights

# ======================== 3. A4 消融模型 =======================
class NoGateFusionModel(nn.Module):
    """
    不含门控融合（Gate Fusion）的完整预测模型。

    该模型为每个IMF分量创建独立的SubNetNoGF子网络进行特征提取和预测，
    将所有IMF的预测结果累加得到最终预测值。这是消融实验中用于验证
    门控融合机制作用的变体版本，使用简单的ConcatFusion替代GateFusion。

    支持"参数共享"模式：share_subnet=True时所有IMF共用同一SubNetNoGF。

    Attributes:
        n_imf (int): IMF分量数量
        n_external (int): 外部特征维度
        share_subnet (bool): 是否共享子网络
        shared_subnet (SubNetNoGF): share_subnet=True时的共享子网
        sub_nets (nn.ModuleList): share_subnet=False时的子网列表
    """

    def __init__(self, n_imf: int, n_external: int,
                 tcn_hidden: int = 32, tcn_output: int = 16,
                 lstm_hidden: int = 32, lstm_layers: int = 2,
                 unified_dim: int = 64, dropout: float = 0.2,
                 share_subnet: bool = True,
                 use_dae: bool = False, dae_pretrained_path: str = None):
        """
        初始化函数。

        Args:
            n_imf (int): IMF分量的数量，即VMD分解后的子序列个数
            n_external (int): 外部特征的维度数量
            tcn_hidden (int, optional): TCN隐藏层维度，默认为32
            tcn_output (int, optional): TCN输出维度，默认为16
            lstm_hidden (int, optional): LSTM隐藏层维度，默认为32
            lstm_layers (int, optional): LSTM层数，默认为2
            unified_dim (int, optional): 融合后的统一维度，默认为64
            dropout (float, optional): Dropout比率，用于正则化防止过拟合，默认为0.2
            share_subnet (bool, optional): 是否所有IMF分量共享同一个子网络，
                True表示共享，False表示每个IMF使用独立子网络，默认为True
            use_dae (bool): 是否使用端到端DAE编码器
            dae_pretrained_path (str): DAE 预训练权重路径
        """
        super(NoGateFusionModel, self).__init__()
        self.n_imf = n_imf
        self.n_external = n_external
        self.share_subnet = share_subnet
        self.use_dae = use_dae

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

        # 构建子网络配置参数字典
        subnet_kwargs = dict(
            input_dim=input_dim,
            tcn_hidden=tcn_hidden,
            tcn_output=tcn_output,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            unified_dim=unified_dim,
            dropout=dropout,
        )

        # 根据share_subnet标志选择共享模式或独立模式创建子网络
        if share_subnet:
            self.shared_subnet = SubNetNoGF(**subnet_kwargs)
        else:
            self.sub_nets = nn.ModuleList([
                SubNetNoGF(**subnet_kwargs) for _ in range(n_imf)
            ])

    def compute_dae_loss(self, raw_externals: torch.Tensor) -> torch.Tensor:
        """
        计算DAE（Denoising Autoencoder）的重构损失。

        该方法用于在端到端训练过程中对DAE编码器进行正则化约束，通过计算重构误差来保证编码质量。
        如果未启用DAE（use_dae=False），则返回零张量以保持损失计算的兼容性。

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
        batch_size, seq_len, n_raw = raw_externals.shape
        ext_flat = raw_externals.reshape(-1, n_raw)
        _, decoded_flat = self.dae(ext_flat, noise=self.training)
        return F.mse_loss(decoded_flat, ext_flat)

    def forward(self, imfs: torch.Tensor, externals: torch.Tensor):
        """
        前向传播函数。

        Args:
            imfs (torch.Tensor): IMF分量张量，形状为 (batch_size, seq_length, n_imf)
            externals (torch.Tensor): 外部特征张量，形状为 (batch_size, seq_length, n_external)

        Returns:
            tuple: (总预测值, 所有注意力权重)
                - total_pred (torch.Tensor): 所有IMF预测结果的累加，形状为 (batch_size,)
                - all_weights (list): 每个IMF子网络的注意力权重列表
        """
        # DAE编码（端到端模式）
        if self.use_dae:
            batch_size, seq_len, n_raw = externals.shape
            ext_flat = externals.reshape(-1, n_raw)
            encoded_flat, _ = self.dae(ext_flat, noise=self.training)
            ext_for_subnet = encoded_flat.reshape(batch_size, seq_len, -1)
        else:
            ext_for_subnet = externals

        total_pred = 0.0
        all_weights = []

        for i in range(self.n_imf):
            imf_i = imfs[:, :, i:i + 1]
            x_i = torch.cat([imf_i, ext_for_subnet], dim=-1)

            if self.share_subnet:
                pred_i, weights_i = self.shared_subnet(x_i)
            else:
                pred_i, weights_i = self.sub_nets[i](x_i)

            total_pred += pred_i
            all_weights.append(weights_i)

        return total_pred, all_weights
