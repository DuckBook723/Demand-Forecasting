"""
文件位置: src/models/fusion.py
作用: 门控融合与预测头
      将CA-TCN提取的局部特征(F_local)与TA-BiLSTM提取的全局特征(F_global)通过可学习门控机制自适应融合，
      再经多层全连接网络输出最终的需求预测值。
所包含的类:
  - GateFusion(nn.Module): 门控融合模块，将局部/全局特征投影到统一维度后，通过Sigmoid门控动态加权融合
  - PredictionHead(nn.Module): 预测头模块，多层全连接网络(BatchNorm + ReLU + Dropout)将融合特征映射为标量预测值
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 被 src/models/vmd_cat_bilstm.py 中的 SubNet 调用
  - 接收来自 CA-TCN 的局部特征和 TA-BiLSTM 的全局特征
  - 输出融合特征供 PredictionHead 生成最终需求预测
  - 本身不直接读写文件，作为端到端模型的一部分参与训练与推理
"""

# ======================== 0. 引入包 =======================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================== 1. 门控融合机制 =======================
class GateFusion(nn.Module):
    """
    门控融合模块（Gate Fusion）

    通过可学习的门控机制自适应地融合局部特征和全局特征。
    门控值控制两种特征的贡献比例，使模型能够动态平衡局部和全局信息。
    """

    def __init__(self, local_dim: int, global_dim: int, unified_dim: int = 64):
        """
        初始化门控融合模块

        Args:
            local_dim: 局部特征的维度
            global_dim: 全局特征的维度
            unified_dim: 统一映射后的维度，默认64
        """
        super(GateFusion, self).__init__()

        # 局部特征投影层：将局部特征映射到统一维度
        self.local_proj = nn.Linear(local_dim, unified_dim)

        # 全局特征投影层：将全局特征映射到统一维度
        self.global_proj = nn.Linear(global_dim, unified_dim)

        # 门控网络：生成0-1范围的门控值，用于控制局部和全局特征的融合比例
        self.gate = nn.Sequential(
            nn.Linear(unified_dim * 2, unified_dim),
            nn.ReLU(inplace=True),
            nn.Linear(unified_dim, unified_dim),
            nn.Sigmoid()
        )

    def forward(self, local_feature: torch.Tensor, global_feature: torch.Tensor):
        """
        前向传播

        Args:
            local_feature: 局部特征张量，形状为 (batch_size, seq_len, local_dim) 或 (batch_size, local_dim)
            global_feature: 全局特征张量，形状为 (batch_size, seq_len, global_dim) 或 (batch_size, global_dim)

        Returns:
            f_fused: 融合后的特征张量，形状为 (batch_size, unified_dim)
        """
        # 如果输入是3维张量，沿时间维度求平均转换为2维
        if local_feature.dim() == 3:
            local_feature = local_feature.mean(dim=1)

        if global_feature.dim() == 3:
            global_feature = global_feature.mean(dim=1)

        # 将局部和全局特征投影到统一维度
        local_unified = self.local_proj(local_feature)
        global_unified = self.global_proj(global_feature)

        # 拼接两种特征并计算门控值
        concat = torch.cat([local_unified, global_unified], dim=1)
        gate = self.gate(concat)

        # 门控融合：根据门控值加权组合局部和全局特征
        f_fused = gate * local_unified + (1 - gate) * global_unified

        return f_fused

# ======================== 2. 预测头 =======================
class PredictionHead(nn.Module):
    """
    预测头模块，用于构建多层感知机进行最终预测。

    该类构建一个包含多个全连接层的前馈神经网络，每层都包含批归一化、
    ReLU激活函数和Dropout正则化，最后输出单一预测值。

    Args:
        input_dim (int): 输入特征的维度
        hidden_dims (list, optional): 隐藏层维度列表，默认为[64, 32]
        dropout (float, optional): Dropout丢弃率，默认为0.2

    Example:
        >>> head = PredictionHead(input_dim=128, hidden_dims=[64, 32], dropout=0.2)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = head(x)  # output shape: (32,)
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.2):
        super().__init__()
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        # 输出层：添加 Sigmoid 将输出压缩到 (0,1)
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())  # ← 关键修改
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)

        Returns:
            torch.Tensor: 预测结果张量，形状为 (batch_size,)
        """
        return self.net(x).squeeze(-1)