"""
文件位置: src/models/vmd_cat_bilstm.py
作用: VMD-CAT-BiLSTM 完整预测模型
      对VMD分解后的各IMF分量独立建模，每个IMF通过共享结构的SubNet（CA-TCN + TA-BiLSTM + 门控融合 + 预测头）进行预测，
      最后将所有IMF的预测值累加得到最终需求预测。
所包含的类:
  - SubNet(nn.Module): 单分量预测子网络，集成CA-TCN局部特征提取、TA-BiLSTM全局时序建模、门控融合与预测头
  - VMDCATBiLSTM(nn.Module): 主模型，管理多个SubNet并行处理各IMF分量，汇总输出最终预测结果及注意力权重
是否需要创建: 是，本文件需要创建
相关文件更新:
  - 依赖 src/models/ca_tcn.py (CATCN)
  - 依赖 src/models/ta_bilstm.py (TABiLSTM)
  - 依赖 src/models/fusion.py (GateFusion, PredictionHead)
  - 运行时从外部接收形状为 (batch, seq_len, n_imf) 的IMF数据与 (batch, seq_len, n_external) 的外部特征
  - 输出最终需求预测值 (batch,) 及各子网的注意力权重列表，用于后续评估与可视化
"""

# ======================== 0. 引入包 =======================
import torch
import torch.nn as nn
from src.models.ca_tcn import CATCN
from src.models.fusion import GateFusion, PredictionHead
from src.models.ta_bilstm import TABiLSTM

# ======================== 1. 独立子网络 =======================
class SubNet(nn.Module):
    """
    VMD-CAT-BiLSTM模型的单分量预测子网络

    该子网络用于对VMD分解后的单个IMF分量进行特征提取和预测，集成了以下模块:
    - CA-TCN: 通道注意力时间卷积网络，用于局部时序特征提取
    - TA-BiLSTM: 时间注意力双向LSTM，用于全局时序依赖建模
    - GateFusion: 门控融合机制，自适应融合局部和全局特征
    - PredictionHead: 多层感知机预测头，输出最终预测结果

    Attributes:
        ca_tcn (CATCN): 通道注意力时间卷积网络，提取局部时序特征
        local_pool (nn.AdaptiveAvgPool1d): 自适应平均池化层，将序列特征压缩为向量
        ta_bilstm (TABiLSTM): 时间注意力双向LSTM，捕捉全局时序依赖关系
        fusion (GateFusion): 门控融合模块，动态融合局部特征和全局特征
        prediction_head (PredictionHead): 预测头网络，将融合特征映射到预测值
    """

    def __init__(
            self,
            input_dim: int,
            tcn_hidden: int = 64,
            tcn_output: int = 32,
            lstm_hidden: int = 64,
            lstm_layers: int = 2,
            unified_dim: int = 128
    ):
        """
        初始化SubNet子网络

        Args:
            input_dim (int): 输入特征维度，通常为1(IMF值) + n_external(外部特征数)
            tcn_hidden (int, optional): TCN隐藏层维度，默认为64
            tcn_output (int, optional): TCN输出特征维度，默认为32
            lstm_hidden (int, optional): LSTM隐藏层维度，默认为64
            lstm_layers (int, optional): LSTM层数，默认为2
            unified_dim (int, optional): 融合后的统一特征维度，默认为128
        """
        super(SubNet, self).__init__()

        self.ca_tcn = CATCN(
            input_dim=input_dim,
            hidden_dim=tcn_hidden,
            output_dim=tcn_output,
        )

        self.local_pool = nn.AdaptiveAvgPool1d(1)

        self.ta_bilstm = TABiLSTM(
            input_dim=tcn_output,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
        )

        self.fusion = GateFusion(
            local_dim=tcn_output,
            global_dim=lstm_hidden * 2,
            unified_dim=unified_dim,
        )

        self.prediction_head = PredictionHead(
            input_dim=unified_dim,
            hidden_dims=[64, 32],
            dropout=0.2,
        )

    def forward(self, x):
        """
        前向传播过程

        处理流程:
        1. 通过CA-TCN提取局部时序特征序列
        2. 对局部特征序列进行全局平均池化，得到局部特征向量
        3. 通过TA-BiLSTM捕捉全局时序依赖，得到全局特征向量和注意力权重
        4. 使用门控融合机制自适应融合局部和全局特征
        5. 通过预测头网络生成最终预测值

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            tuple:
                - y_pred (torch.Tensor): 预测值，形状为 (batch_size,)
                - atten_weights (torch.Tensor): TA-BiLSTM的注意力权重，用于可视化分析
        """
        f_local_seq = self.ca_tcn(x)
        f_local = self.local_pool(f_local_seq.transpose(1, 2)).squeeze(-1)
        f_global, atten_weights = self.ta_bilstm(f_local_seq)
        f_fused = self.fusion(f_local, f_global)
        y_pred = self.prediction_head(f_fused)
        return y_pred, atten_weights

# ======================== 2. 完整模型VMDCATBiLSTM =======================
class VMDCATBiLSTM(nn.Module):
    """
    VMD-CAT-BiLSTM主预测模型

    该模型基于VMD(变分模态分解)方法，将原始时间序列分解为多个IMF(本征模态函数)分量，
    为每个IMF分量独立分配一个SubNet子网络进行并行预测，最后将所有子网络的预测结果
    累加得到最终的需求预测值。同时收集各子网络的注意力权重，用于模型可解释性分析。

    模型架构:
        输入IMFs → [SubNet_1, SubNet_2, ..., SubNet_n] → 预测值累加 → 输出

    Attributes:
        n_imf (int): IMF分量的数量，即VMD分解后的模态数
        n_external (int): 外部特征的维度数量
        sub_nets (nn.ModuleList): 包含n_imf个SubNet子网络的模块列表
    """

    def __init__(
            self,
            n_imf: int,
            n_external: int,
            tcn_hidden: int = 64,
            tcn_output: int = 32,
            lstm_hidden: int = 64,
            lstm_layers: int = 2,
            unified_dim: int = 128
    ):
        """
        初始化VMDCATBiLSTM主模型

        为每个IMF分量创建独立的SubNet子网络，所有子网络共享相同的网络结构和超参数配置。

        Args:
            n_imf (int): IMF分量数量，决定创建多少个子网络
            n_external (int): 外部特征的数量，如温度、湿度、节假日标记等
            tcn_hidden (int, optional): TCN隐藏层维度，默认为64
            tcn_output (int, optional): TCN输出维度，默认为32
            lstm_hidden (int, optional): BiLSTM隐藏层维度，默认为64
            lstm_layers (int, optional): BiLSTM层数，默认为2
            unified_dim (int, optional): 融合特征的统一维度，默认为128
        """
        super(VMDCATBiLSTM, self).__init__()

        self.n_imf = n_imf
        self.n_external = n_external

        input_dim = 1 + n_external

        # 为每个IMF分量创建独立的SubNet子网络
        self.sub_nets = nn.ModuleList([
            SubNet(
                input_dim=input_dim,
                tcn_hidden=tcn_hidden,
                tcn_output=tcn_output,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                unified_dim=unified_dim
            )
            for _ in range(n_imf)
        ])

    def forward(self, imfs: torch.Tensor, externals: torch.Tensor):
        """
        前向传播过程

        处理流程:
        1. 遍历每个IMF分量，提取对应的IMF数据
        2. 将当前IMF与外部特征在特征维度拼接
        3. 通过对应的SubNet子网络进行预测，获取预测值和注意力权重
        4. 累加所有IMF的预测值得到最终预测结果
        5. 收集所有子网络的注意力权重用于后续分析

        Args:
            imfs (torch.Tensor): IMF分量张量，形状为 (batch_size, seq_len, n_imf)
            externals (torch.Tensor): 外部特征张量，形状为 (batch_size, seq_len, n_external)

        Returns:
            tuple:
                - total_pred (torch.Tensor): 最终预测值，形状为 (batch_size,)，为所有IMF预测值的累加和
                - all_weights (list[torch.Tensor]): 注意力权重列表，长度为n_imf，
                  每个元素为对应SubNet的注意力权重，用于可视化各IMF的时间重要性分布
        """
        total_pred = 0.0
        all_weights = []

        # 遍历每个IMF分量，独立预测后累加
        for i in range(self.n_imf):
            imf_i = imfs[:, :, i:i + 1]
            x_i = torch.cat([imf_i, externals], dim=-1)
            pred_i, weights_i = self.sub_nets[i](x_i)
            total_pred += pred_i
            all_weights.append(weights_i)

        return total_pred, all_weights