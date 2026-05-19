"""
文件位置: experiments/ablation/no_ta.py
作用: 消融实验模型 A4 —— 去除时间注意力机制(TA)的对比模型
      完整保留了 VMD-CAT-BiLSTM 的 VMD 分解、CA-TCN 局部特征提取、双向 LSTM、
      门控融合和预测头架构，但将 TA-BiLSTM 中的 TemporalAttention 模块移除，
      替换为直接取 BiLSTM 最后一个时间步隐藏状态的简单聚合策略。
      用于验证时间注意力对模型性能的贡献：若该模型指标明显差于完整模型，则证明时间注意力有效。
      支持端到端DAE训练：当use_dae=True时，DAE编码器在forward中编码原始特征。
所包含的类:
  - BiLSTMNoTA(nn.Module): 无时间注意力的双向LSTM，取末步隐藏状态作为全局特征，返回均匀假权重
  - SubNetNoTA(nn.Module): 无TA的单分量预测子网络
  - NoTAModel(nn.Module): 无时间注意力的完整消融模型
是否需要创建: 是，本文件需要创建（P3 消融实验 A4 专用）
相关文件更新:
  - 依赖 src/models/ca_tcn.py (CATCN)
  - 依赖 src/models/fusion.py (GateFusion, PredictionHead)
  - 不依赖 src/models/ta_bilstm.py (TABiLSTM)，因为时间注意力已被移除
  - 输入数据与完整模型相同：IMF 分量 + DAE 编码外部特征
  - 输出预测值与伪注意力权重（仅用于接口兼容，无实际物理意义），供 P3 实验评估使用
"""

# ======================== 0. 引入包 =======================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ca_tcn import CATCN
from src.models.fusion import GateFusion, PredictionHead
from src.features.dae_encoder import DenoisingAE
from configs.config import (
    DAE_HIDDEN_DIM,             # DAE 隐藏层维度
    NOISE_STD,                  # DAE 噪声标准差
    DAE_MODEL_FILE_PATH,        # DAE 模型文件路径
    DROPOUT,                    # dropout 概率
)

# ======================== 1. 无时间注意力的BiLSTM =======================
class BiLSTMNoTA(nn.Module):
    """
    不含时间注意力（Temporal Attention）的双向LSTM模块。

    该模块通过输入投影层、双向LSTM和层归一化提取时序特征，
    使用最后一个时间步的输出作为全局特征表示，而非注意力加权。
    生成均匀的虚拟注意力权重仅用于接口兼容。
    这是消融实验中用于验证时间注意力机制作用的变体版本。

    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int, optional): LSTM隐藏层维度，默认为64
        num_layers (int, optional): LSTM层数，默认为2

    Example:
        >>> model = BiLSTMNoTA(input_dim=32, hidden_dim=64, num_layers=2)
        >>> x = torch.randn(32, 100, 32)  # batch_size=32, seq_len=100
        >>> last_hidden, dummy_weights = model(x)
        # last_hidden shape: (32, 128), dummy_weights shape: (32, 100)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(BiLSTMNoTA, self).__init__()

        # 输入投影：将原始特征映射到隐藏维度
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # 双向LSTM编码时序信息
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 层归一化和Dropout正则化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, input_dim)

        Returns:
            tuple: (最后时间步输出, 虚拟注意力权重)
                - last_hidden (torch.Tensor): 最后一个时间步的双向LSTM输出，形状为 (batch_size, hidden_dim * 2)
                - dummy_weights (torch.Tensor): 均匀分布的虚拟注意力权重，形状为 (batch_size, seq_length)
        """
        x = self.input_proj(x)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.drop(lstm_out)

        # 取前向LSTM末步 + 反向LSTM首步的拼接（正确利用双向信息）
        # 前向LSTM的末步已看过全部序列，反向LSTM的首步也已看过全部序列
        forward_last = lstm_out[:, -1, :lstm_out.size(-1) // 2]  # 前向末步
        backward_first = lstm_out[:, 0, lstm_out.size(-1) // 2:]   # 反向首步
        last_hidden = torch.cat([forward_last, backward_first], dim=-1)

        # 生成均匀注意力权重（仅用于接口兼容，不影响模型计算）
        batch_size, seq_len, _ = lstm_out.shape
        dummy_weights = torch.ones(batch_size, seq_len, device=x.device) / seq_len

        return last_hidden, dummy_weights

# ======================== 2. 无时间注意力的SubNet =======================
class SubNetNoTA(nn.Module):
    """
    不含时间注意力（Temporal Attention）的子网络模块。

    该网络结合了带通道注意力的TCN和无时间注意力的BiLSTM，通过门控融合机制整合局部特征和全局特征，
    最后通过预测头输出预测结果。这是消融实验中用于验证时间注意力机制作用的变体版本。

    Args:
        input_dim (int): 输入特征维度
        tcn_hidden (int, optional): TCN隐藏层维度，默认为64
        tcn_output (int, optional): TCN输出维度，默认为32
        lstm_hidden (int, optional): LSTM隐藏层维度，默认为64
        lstm_layers (int, optional): LSTM层数，默认为2
        unified_dim (int, optional): 融合后的统一维度，默认为128

    Example:
        >>> model = SubNetNoTA(input_dim=10, tcn_hidden=64, tcn_output=32,
        ...                    lstm_hidden=64, lstm_layers=2, unified_dim=128)
        >>> x = torch.randn(32, 100, 10)  # batch_size=32, seq_len=100
        >>> y_pred, atten_weights = model(x)
    """

    def __init__(self, input_dim: int, tcn_hidden: int = 64, tcn_output: int = 32,
                 lstm_hidden: int = 64, lstm_layers: int = 2, unified_dim: int = 128, dropout: float = 0.2):
        super(SubNetNoTA, self).__init__()

        # 使用带通道注意力的TCN提取局部时序特征
        self.ca_tcn = CATCN(
            input_dim=input_dim,
            hidden_dim=tcn_hidden,
            output_dim=tcn_output,
            dropout=dropout,
        )

        # 全局池化获取序列级特征
        self.local_pool = nn.AdaptiveAvgPool1d(1)

        # 使用无时间注意力的BiLSTM替代TABiLSTM
        self.bilstm = BiLSTMNoTA(
            input_dim=tcn_output,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # 门控融合机制整合局部和全局特征
        self.fusion = GateFusion(
            local_dim=tcn_output,
            global_dim=lstm_hidden * 2,
            unified_dim=unified_dim,
        )

        # 预测头生成最终预测结果
        self.prediction_head = PredictionHead(
            input_dim=unified_dim,
            hidden_dims=[unified_dim, unified_dim // 2],
            dropout=dropout,
        )

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, input_dim)

        Returns:
            tuple: (预测值, 虚拟注意力权重)
                - y_pred (torch.Tensor): 预测结果，形状为 (batch_size,)
                - atten_weights (torch.Tensor): 虚拟的均匀注意力权重
        """
        f_local_seq = self.ca_tcn(x)
        f_local = self.local_pool(f_local_seq.transpose(1, 2)).squeeze(-1)
        f_global, atten_weights = self.bilstm(f_local_seq)
        f_fused = self.fusion(f_local, f_global)
        y_pred = self.prediction_head(f_fused)
        return y_pred, atten_weights

# ======================== 3. A3 消融模型 =======================
class NoTAModel(nn.Module):
    """
    不含时间注意力（Temporal Attention）的完整消融模型。

    该模型是 VMD-CAT-BiLSTM 的消融实验变体 A4，移除了时间注意力机制以验证其对模型性能的贡献。
    模型保留 VMD 分解、CA-TCN 局部特征提取、BiLSTM 全局特征提取、门控融合和预测头架构，
    但将 TA-BiLSTM 替换为普通的 BiLSTM，直接取最后一个时间步的隐藏状态作为全局特征。

    模型对每个 IMF 分量使用独立的子网络（或共享子网络）进行预测，最后累加所有 IMF 的预测结果。

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
        dae_pretrained_path (str): DAE 预训练权重文件路径
    Example:
        >>> model = NoTAModel(n_imf=5, n_external=10, share_subnet=True)
        >>> imfs = torch.randn(32, 100, 5)  # batch_size=32, seq_len=100
        >>> externals = torch.randn(32, 100, 10)
        >>> total_pred, all_weights = model(imfs, externals)
        # total_pred shape: (32,), all_weights: list of 5 tensors each with shape (32, 100)
    """

    def __init__(self, n_imf: int, n_external: int,
                 tcn_hidden: int = 32, tcn_output: int = 16,
                 lstm_hidden: int = 32, lstm_layers: int = 2,
                 unified_dim: int = 64, dropout: float = 0.2,
                 share_subnet: bool = True,
                 use_dae: bool = False, dae_pretrained_path: str = None):
        super(NoTAModel, self).__init__()
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

        subnet_kwargs = dict(
            input_dim=input_dim,
            tcn_hidden=tcn_hidden,
            tcn_output=tcn_output,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            unified_dim=unified_dim,
            dropout=dropout,
        )

        if share_subnet:
            self.shared_subnet = SubNetNoTA(**subnet_kwargs)
        else:
            self.sub_nets = nn.ModuleList([
                SubNetNoTA(**subnet_kwargs) for _ in range(n_imf)
            ])

    def compute_dae_loss(self, raw_externals: torch.Tensor) -> torch.Tensor:
        """
        计算DAE（Denoising Autoencoder）的重构损失。

        该方法用于在端到端训练过程中对DAE编码器进行正则化约束，通过计算重构误差来保证编码质量。
        如果未启用DAE（use_dae=False），则返回零张量以保持损失计算的兼容性。

        优先使用 forward() 中缓存的解码结果，避免重复前向传播导致噪声不一致。
        使用 _cache_valid 标志位判断缓存是否来自当前 forward 调用，
        避免依赖 Python 对象同一性（is 检查），因为 PyTorch 切片操作每次创建新视图对象。

        Args:
            raw_externals (torch.Tensor): 原始外部特征张量，形状为 (batch_size, seq_length, n_external)

        Returns:
            torch.Tensor: DAE重构损失标量值。如果未启用DAE则返回0.0，否则返回均方误差损失
        """
        if not self.use_dae:
            return torch.tensor(0.0, device=raw_externals.device)
        # 优先使用 forward() 中缓存的解码结果，避免重复前向传播
        # 注意：不能用 is 检查对象同一性，因为 PyTorch 切片操作每次创建新视图对象
        if getattr(self, '_cache_valid', False):
            if self._cached_decoded.shape == raw_externals.shape:
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

    def forward(self, imfs: torch.Tensor, externals: torch.Tensor):
        """
        前向传播函数。

        Args:
            imfs (torch.Tensor): IMF分量张量，形状为 (batch_size, seq_length, n_imf)
            externals (torch.Tensor): 外部特征张量，形状为 (batch_size, seq_length, n_external)

        Returns:
            tuple: (总预测值, 所有虚拟注意力权重)
                - total_pred (torch.Tensor): 所有IMF预测结果的累加，形状为 (batch_size,)
                - all_weights (list): 每个IMF子网络的虚拟注意力权重列表
        """
        # DAE编码（端到端模式）
        if self.use_dae:
            batch_size, seq_len, n_raw = externals.shape
            ext_flat = externals.reshape(-1, n_raw)
            encoded_flat, decoded_flat = self.dae(ext_flat, noise=self.training)
            # 缓存解码结果，供 compute_dae_loss() 使用，避免重复前向传播
            self._cached_decoded = decoded_flat.reshape(batch_size, seq_len, n_raw)
            self._cached_raw = externals
            self._cache_valid = True  # 标记缓存有效（当前 forward 调用产生）
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