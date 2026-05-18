"""
文件位置: src/models/ta_bilstm.py
作用: 时间注意力双向LSTM(TA-BiLSTM)——全局时序建模
所包含的类:
  - TemporalAttention(nn.Module): 时间注意力机制，为每个时间步生成权重
  - TABiLSTM(nn.Module): TA-BiLSTM主模块，结合BiLSTM与时间注意力
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 读取 data/processed/*_encoded.csv (由外部数据加载器提供)
  - 输出 无 (模块作为主模型的一部分，不单独保存)
"""

# ======================== 0. 引入包 =======================
import torch.nn as nn
import torch

# ======================== 1. 时间注意力 =======================
class TemporalAttention(nn.Module):
    """
    时间注意力机制（Temporal Attention）

    对时间序列的每个时间步计算注意力权重，使模型能够关注重要的时间步。
    通过两层全连接网络计算注意力分数，并使用softmax归一化为权重。
    """
    def __init__(self,hidden_dim: int):
        """
        初始化时间注意力模块

        Args:
            hidden_dim: 隐藏层维度，必须为偶数
        """
        super(TemporalAttention, self).__init__()
        assert hidden_dim % 2 == 0, "hidden_dim should be even for TemporalAttention"
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_dim)

        Returns:
            context: 上下文向量，形状为 (batch_size, hidden_dim)，加权求和后的时间特征
            weights: 注意力权重，形状为 (batch_size, seq_len)，表示每个时间步的重要性
        """
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)

        context = torch.bmm(
            weights.unsqueeze(1),
            x
        ).squeeze(1)

        return context, weights

# ======================== 2. TABiLSTM =======================
class TABiLSTM(nn.Module):
    """
    时间注意力双向LSTM（Temporal Attention BiLSTM）

    结合双向LSTM和时间注意力机制的模型，用于捕获时间序列的双向依赖关系，
    并通过注意力机制自动识别关键时间步，适用于需求预测等时序任务。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        初始化TA-BiLSTM模型

        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度，默认64
            num_layers: LSTM层数，默认2
            dropout: Dropout比率，默认0.2
        """
        super(TABiLSTM, self).__init__()

        # 输入投影层：将原始输入映射到隐藏层维度
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # 双向LSTM：同时捕获前向和后向的时间依赖关系
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout  if num_layers > 1 else 0.0
        )

        # 层归一化：对双向LSTM的输出进行归一化（维度为hidden_dim*2）
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # 时间注意力模块：对LSTM输出的时间步进行加权
        self.temporal_attention = TemporalAttention(hidden_dim * 2)

        # Dropout层：防止过拟合
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            context: 上下文向量，形状为 (batch_size, hidden_dim*2)，经注意力加权后的时序特征
            weights: 注意力权重，形状为 (batch_size, seq_len)，表示每个时间步的重要性
        """
        # 将输入投影到隐藏层维度
        x = self.input_proj(x)

        # 双向LSTM编码：输出维度为hidden_dim*2（拼接前向和后向隐藏状态）
        lstm_out, _ = self.bilstm(x)

        # 层归一化：稳定训练过程
        lstm_out = self.layer_norm(lstm_out)

        # Dropout正则化
        lstm_out = self.drop(lstm_out)

        # 时间注意力：计算加权的上下文向量和注意力权重
        context, weights = self.temporal_attention(lstm_out)

        return context, weights