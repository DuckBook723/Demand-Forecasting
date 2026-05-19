"""
文件位置: src/models/ca_tcn.py
作用: 通道注意力时域卷积网络(CA-TCN)——局部时序特征提取
所包含的类:
  - CausalConv1d(nn.Module): 因果一维卷积，保证时序因果性
  - ECAAttention(nn.Module): 高效通道注意力(ECA)，动态调整通道重要性
  - TemporalBlock(nn.Module): TCN 残差块，含膨胀因果卷积与通道注意力
  - CATCN(nn.Module): CA-TCN 主模块，堆叠多个 TemporalBlock
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 读取 data/processed/*_encoded.csv (集成时由外部加载)
  - 输出 checkpoints/ca_tcn.pth (可独立保存或随主模型保存)
"""

# ======================== 0. 引入包 =======================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================== 1. 因果空洞卷积 =======================
class CausalConv1d(nn.Module):
    """
    因果一维卷积层（Causal Conv1d）

    实现因果卷积，确保输出仅依赖于当前及过去的输入，不会"看到"未来的信息。
    通过左侧填充和右侧裁剪来实现因果性，常用于时间序列预测任务。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,stride: int=1, dilation: int=1):
        """
        初始化因果卷积层

        Args:
           in_channels: 输入通道数
           out_channels: 输出通道数
           kernel_size: 卷积核大小
           stride: 卷积步长，目前仅支持1
           dilation: 膨胀系数，控制卷积核的间距
        """
        super(CausalConv1d, self).__init__()

        assert stride == 1, "stride != 1 not supported"

        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
            stride=stride,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

# ======================== 2. ECAAttention 通道注意力 =======================
class ECAAttention(nn.Module):
    """
    高效通道注意力模块（Efficient Channel Attention, ECA）

    通过自适应平均池化和一维卷积来捕获通道间的依赖关系，
    在不引入全连接层的情况下实现通道注意力机制，计算效率高。
    """
    def __init__(self, kernel_size: int = 5):
        """
        初始化ECA注意力模块

        Args:
            channels: 输入特征图的通道数
            kernel_size: 一维卷积的核大小，用于捕获局部跨通道交互，默认5
        """
        super(ECAAttention, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, channels, seq_len)

        Returns:
            应用通道注意力加权后的输出张量，形状与输入相同
        """
        y = self.avg_pool(x).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# ======================== 3. 时序模块=======================
class TemporalBlock(nn.Module):
    """
    时间块（Temporal Block）

    TCN的核心构建模块，包含两个因果卷积层组成的残差块。
    每个卷积层后跟随批归一化、ReLU激活和Dropout正则化。
    通过残差连接实现梯度流动，支持深层网络训练。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2, eca_kernel_size: int = 5):
        """
        初始化时间块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dilation: 膨胀系数，用于扩大感受野
            dropout: Dropout比率，用于防止过拟合，默认0.2
        """
        super(TemporalBlock, self).__init__()

        self.layer1 = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer2 = nn.Sequential(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.net = nn.Sequential(
            self.layer1,
            self.layer2,
        )

        self.eca = ECAAttention(eca_kernel_size)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, channels, seq_len)

        Returns:
            经过残差连接和激活函数后的输出张量
        """
        out = self.net(x)
        res = self.residual_conv(x)
        out = out + res
        out = F.relu(out)
        out = self.eca(out)
        return out

# ======================== 4. CATCN 模块 =======================
class CATCN(nn.Module):
    """
    时间卷积网络块（Temporal Convolutional Network Block）

    由多个TemporalBlock堆叠而成，每层使用指数增长的膨胀系数（2^i），
    使网络能够捕获不同时间尺度的依赖关系，同时保持较大的感受野。
    """
    def __init__(self, input_dim: int, hidden_dim: int=64, output_dim: int=32, num_layers: int=4, kernel_size: int=3, dropout: float=0.2):

        super(CATCN, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout))

        self.net = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, channels, seq_len)

        Returns:
            经过TCN处理后的输出张量
        """
        x = self.input_proj(x).transpose(1, 2)
        x = self.net(x).transpose(1, 2)
        f_local = self.output_proj(x)
        return f_local

# ======================== 测试 =======================
if __name__ == '__main__':
    pass
