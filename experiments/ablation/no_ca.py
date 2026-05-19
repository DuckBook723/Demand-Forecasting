"""
文件位置: experiments/ablation/no_ca.py
作用: 消融实验模型 A5 —— 去除通道注意力机制(ECA)的对比模型
      完整保留了 VMD-CAT-BiLSTM 的 VMD 分解、双分支并行、因果空洞卷积、残差连接、
      TA-BiLSTM、门控融合和预测头架构，但将 CA-TCN 中的 ECA-Net 注意力模块移除，
      替换为仅含因果空洞卷积与残差连接的普通 TCN。
      用于验证通道注意力对模型性能的贡献：若该模型指标明显差于完整模型，则证明通道注意力有效。
      支持端到端DAE训练：当use_dae=True时，DAE编码器在forward中编码原始特征。
所包含的类:
  - TemporalBlockNoCA(nn.Module): 无ECA的TCN残差块（因果空洞卷积 + 残差连接）
  - TCNNoCA(nn.Module): 无通道注意力的时域卷积网络
  - SubNetNoCA(nn.Module): 无ECA的单分量预测子网络
  - NoCAModel(nn.Module): 无通道注意力的完整消融模型
是否需要创建: 是，本文件需要创建（P3 消融实验 A5 专用）
相关文件更新:
  - 依赖 src/models/ca_tcn.py (仅引用 CausalConv1d 基类)
  - 依赖 src/models/ta_bilstm.py (TABiLSTM)
  - 依赖 src/models/fusion.py (GateFusion, PredictionHead)
  - 输入数据与完整模型相同：IMF 分量 + DAE 编码外部特征
  - 输出预测值与注意力权重，供 P3 实验评估使用
"""

# ======================== 0. 引入包 =======================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ca_tcn import CausalConv1d
from src.models.ta_bilstm import TABiLSTM
from src.models.fusion import GateFusion, PredictionHead
from src.features.dae_encoder import DenoisingAE
from configs.config import (
    DAE_HIDDEN_DIM,             # DAE 隐藏层维度
    NOISE_STD,                  # DAE 噪声标准差
    DAE_MODEL_FILE_PATH,        # DAE 模型文件路径
    DROPOUT,                    # dropout 概率
)

# ======================== 1. 无通道注意力的TemporalBlock =======================
class TemporalBlockNoCA(nn.Module):
    """
    不含通道注意力（Channel Attention）的时间卷积块。

    该模块由两个因果卷积层组成，每层包含批归一化、ReLU激活和Dropout正则化。
    使用残差连接将输入与输出相加，如果输入输出通道数不一致则通过1x1卷积进行维度匹配。
    这是消融实验中用于验证通道注意力机制作用的变体版本。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        dilation (int): 空洞卷积的扩张率
        dropout (float, optional): Dropout丢弃率，默认为0.2

    Example:
        >>> block = TemporalBlockNoCA(in_channels=64, out_channels=128, kernel_size=3, dilation=2)
        >>> x = torch.randn(32, 64, 100)  # batch_size=32, seq_len=100
        >>> output = block(x)  # output shape: (32, 128, 100)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super(TemporalBlockNoCA, self).__init__()

        # 第一层因果卷积模块
        self.layer1 = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 第二层因果卷积模块
        self.layer2 = nn.Sequential(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.net = nn.Sequential(self.layer1, self.layer2)

        # 残差连接的维度匹配层
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, seq_length)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, seq_length)
        """
        out = self.net(x)
        # 不使用ECA，直接加残差
        res = self.residual_conv(x)
        out = out + res
        return F.relu(out)

# ======================== 2. 无通道注意力的TCN =======================
class TCNNoCA(nn.Module):
    """
    不含通道注意力（Channel Attention）的时间卷积网络。

    该网络由输入投影层、多个TemporalBlockNoCA层和输出投影层组成。
    使用指数扩张率的空洞因果卷积捕捉不同时间尺度的依赖关系。
    这是消融实验中用于验证通道注意力机制作用的变体版本。

    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int, optional): 隐藏层维度，默认为64
        output_dim (int, optional): 输出维度，默认为32
        num_layers (int, optional): TCN层数，默认为4
        kernel_size (int, optional): 卷积核大小，默认为3
        dropout (float, optional): Dropout丢弃率，默认为0.2

    Example:
        >>> model = TCNNoCA(input_dim=10, hidden_dim=64, output_dim=32, num_layers=4)
        >>> x = torch.randn(32, 100, 10)  # batch_size=32, seq_len=100
        >>> output = model(x)  # output shape: (32, 100, 32)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
                 num_layers: int = 4, kernel_size: int = 3, dropout: float = 0.2):
        super(TCNNoCA, self).__init__()

        # 输入投影：将原始特征映射到隐藏维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 构建多层TCN模块，每层使用递增的空洞扩张率
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlockNoCA(hidden_dim, hidden_dim, kernel_size, dilation, dropout))

        self.net = nn.Sequential(*layers)

        # 输出投影：将隐藏维度映射到目标维度
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, input_dim)

        Returns:
            torch.Tensor: 局部特征表示，形状为 (batch_size, seq_length, output_dim)
        """
        x = self.input_proj(x).transpose(1, 2)
        x = self.net(x).transpose(1, 2)
        f_local = self.output_proj(x)
        return f_local

# ======================== 3. 无通道注意力的SubNet =======================
class SubNetNoCA(nn.Module):
    """
    不含通道注意力（Channel Attention）的子网络模块。

    该网络结合了无CA的TCN和带时间注意力的BiLSTM，通过门控融合机制整合局部特征和全局特征，
    最后通过预测头输出预测结果。这是消融实验中用于验证通道注意力机制作用的变体版本。

    Args:
        input_dim (int): 输入特征维度
        tcn_hidden (int, optional): TCN隐藏层维度，默认为64
        tcn_output (int, optional): TCN输出维度，默认为32
        lstm_hidden (int, optional): LSTM隐藏层维度，默认为64
        lstm_layers (int, optional): LSTM层数，默认为2
        unified_dim (int, optional): 融合后的统一维度，默认为128

    Example:
        >>> model = SubNetNoCA(input_dim=10, tcn_hidden=64, tcn_output=32,
        ...                    lstm_hidden=64, lstm_layers=2, unified_dim=128)
        >>> x = torch.randn(32, 100, 10)  # batch_size=32, seq_len=100
        >>> y_pred, atten_weights = model(x)
    """

    def __init__(self, input_dim: int, tcn_hidden: int = 64, tcn_output: int = 32,
                 lstm_hidden: int = 64, lstm_layers: int = 2, unified_dim: int = 128,dropout: float = 0.2):
        super(SubNetNoCA, self).__init__()

        # 使用无CA的TCN提取局部时序特征
        self.tcn = TCNNoCA(
            input_dim=input_dim,
            hidden_dim=tcn_hidden,
            output_dim=tcn_output,
            dropout=dropout
        )

        # 全局池化获取序列级特征
        self.local_pool = nn.AdaptiveAvgPool1d(1)

        # 带时间注意力的BiLSTM提取全局时序特征
        self.ta_bilstm = TABiLSTM(
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
            tuple: (预测值, 注意力权重)
                - y_pred (torch.Tensor): 预测结果，形状为 (batch_size,)
                - atten_weights (torch.Tensor): 时间注意力权重
        """
        f_local_seq = self.tcn(x)
        f_local = self.local_pool(f_local_seq.transpose(1, 2)).squeeze(-1)
        f_global, atten_weights = self.ta_bilstm(f_local_seq)
        f_fused = self.fusion(f_local, f_global)
        y_pred = self.prediction_head(f_fused)
        return y_pred, atten_weights

# ======================== 4. A2 消融模型 =======================
class NoCAModel(nn.Module):
    """
    不含通道注意力（Channel Attention）的完整预测模型。

    该模型为每个IMF分量创建独立的SubNetNoCA子网络进行特征提取和预测，
    将所有IMF的预测结果累加得到最终预测值。这是消融实验中用于验证
    通道注意力机制作用的变体版本。
    支持"参数共享"模式和"端到端DAE训练"模式。

    Attributes:
        n_imf (int): IMF分量数量
        n_external (int): 外部特征维度
        shared_subnet (SubNetNoCA): share_subnet=True时的共享子网
        sub_nets (nn.ModuleList): share_subnet=False时的子网列表
        dae (DenoisingAE): DAE编码器+解码器（仅use_dae=True时存在）
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
            share_subnet (bool, optional): 是否所有IMF分量共享同一个子网络
            use_dae (bool): 是否使用端到端DAE编码器
            dae_pretrained_path (str): DAE 预训练权重路径
        """
        super(NoCAModel, self).__init__()
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
            self.shared_subnet = SubNetNoCA(**subnet_kwargs)
        else:
            self.sub_nets = nn.ModuleList([
                SubNetNoCA(**subnet_kwargs) for _ in range(n_imf)
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
            tuple: (总预测值, 所有注意力权重)
                - total_pred (torch.Tensor): 所有IMF预测结果的累加，形状为 (batch_size,)
                - all_weights (list): 每个IMF子网络的注意力权重列表
        """
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

        # 对每个IMF分量独立处理并累加预测结果
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