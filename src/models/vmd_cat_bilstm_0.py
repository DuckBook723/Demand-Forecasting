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
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ca_tcn import CATCN
from src.models.fusion import GateFusion, PredictionHead
from src.models.ta_bilstm import TABiLSTM
from src.features.dae_encoder import DenoisingAE
from configs.config import (
    DAE_HIDDEN_DIM,                 # DAE隐藏层维度
    NOISE_STD,                      # 噪声标准差
    DAE_MODEL_FILE_PATH,            # DAE模型保存路径
    DROPOUT,                        # dropout 概率
)

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
            unified_dim: int = 128,
            dropout: float = 0.2
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
            dropout (float, optional): Dropout比率，默认为0.2
        """
        super(SubNet, self).__init__()

        self.ca_tcn = CATCN(
            input_dim=input_dim,
            hidden_dim=tcn_hidden,
            output_dim=tcn_output,
            dropout=dropout,
        )

        self.local_pool = nn.AdaptiveAvgPool1d(1)

        self.ta_bilstm = TABiLSTM(
            input_dim=tcn_output,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        self.fusion = GateFusion(
            local_dim=tcn_output,
            global_dim=lstm_hidden * 2,
            unified_dim=unified_dim,
        )

        self.prediction_head = PredictionHead(
            input_dim=unified_dim,
            hidden_dims=[unified_dim, unified_dim // 2],
            dropout=dropout,
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
    为每个IMF分量分配SubNet子网络进行预测，最后将所有子网络的预测结果
    累加得到最终的需求预测值。同时收集各子网络的注意力权重，用于模型可解释性分析。

    支持"参数共享"模式：当share_subnet=True时，所有IMF分量共用同一个SubNet，
    参数量降为原来的1/n_imf，且每个参数获得n_imf倍的梯度更新，
    有效缓解小样本下的过拟合问题。

    支持"端到端DAE训练"模式：当use_dae=True时，DAE编码器作为模型组件，
    在forward中将原始n_external维特征编码为dae_output_dim维，然后送入SubNet处理。
    DAE权重根据预测损失+重构损失联合优化，使编码特征对预测任务更有利。

    模型架构:
        输入IMFs → [SubNet_1, SubNet_2, ..., SubNet_n] → 预测值累加 → 输出
        或（share_subnet=True）: 输入IMFs → [SharedSubNet] × n → 预测值累加 → 输出
        或（use_dae=True）: 输入IMFs + 原始特征 → DAE编码 → 编码特征 → SubNet → 输出

    Attributes:
        n_imf (int): IMF分量的数量，即VMD分解后的模态数
        n_external (int): 外部特征的维度数量（原始特征维度）
        use_dae (bool): 是否使用端到端DAE编码器
        dae (DenoisingAE): DAE编码器+解码器（仅use_dae=True时存在）
        sub_nets (nn.ModuleList): share_subnet=False时，包含n_imf个SubNet
        shared_subnet (SubNet): share_subnet=True时，单个共享SubNet
    """
    def __init__(
            self,
            n_imf: int,
            n_external: int,
            tcn_hidden: int = 32,
            tcn_output: int = 16,
            lstm_hidden: int = 32,
            lstm_layers: int = 2,
            unified_dim: int = 64,
            dropout: float = 0.2,
            share_subnet: bool = True,
            use_dae: bool = False,
            dae_pretrained_path: str = None,
    ):
        """
        初始化VMDCATBiLSTM主模型

        Args:
            n_imf (int): IMF分量数量
            n_external (int): 外部特征数量
            tcn_hidden (int): TCN隐藏层维度
            tcn_output (int): TCN输出维度
            lstm_hidden (int): BiLSTM隐藏层维度
            lstm_layers (int): BiLSTM层数
            unified_dim (int): 融合特征统一维度
            dropout (float): Dropout比率
            share_subnet (bool): 是否共享SubNet参数，True时所有IMF共用同一SubNet
            use_dae (bool): 是否使用端到端DAE编码器。True时，模型内部将n_external维原始特征编码为
                            dae_output_dim维，SubNet输入维度=1+dae_output_dim
            dae_pretrained_path (str): DAE预训练权重路径，None则使用config默认路径
        """
        super(VMDCATBiLSTM, self).__init__()

        self.n_imf = n_imf
        self.n_external = n_external
        self.share_subnet = share_subnet
        self.use_dae = use_dae

        # ---- DAE编码器（端到端训练模式） ----
        if use_dae:
            dae_output_dim = DAE_HIDDEN_DIM[-1]  # 编码后特征维度，如3
            self.dae = DenoisingAE(
                input_dim=n_external,
                hidden_dim=DAE_HIDDEN_DIM,
                noise_std=NOISE_STD,
                dropout=dropout,
            )
            # 加载预训练权重作为初始化（如果有）
            pretrained = dae_pretrained_path or DAE_MODEL_FILE_PATH
            if os.path.exists(pretrained):
                state_dict = torch.load(pretrained, map_location='cpu')
                self.dae.load_state_dict(state_dict)
                print(f"  📦 DAE预训练权重已加载: {pretrained}")
            else:
                print(f"  ⚠️ DAE预训练权重未找到，使用随机初始化: {pretrained}")

            self.dae_output_dim = dae_output_dim
            input_dim = 1 + dae_output_dim  # SubNet输入: 1(IMF) + dae_output_dim
        else:
            input_dim = 1 + n_external  # SubNet输入: 1(IMF) + n_external

        if share_subnet:
            # 参数共享：所有IMF共用同一个SubNet，参数量仅为独立模式的1/n_imf
            self.shared_subnet = SubNet(
                input_dim=input_dim,
                tcn_hidden=tcn_hidden,
                tcn_output=tcn_output,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                unified_dim=unified_dim,
                dropout=dropout,
            )
        else:
            # 独立模式：每个IMF分量拥有独立的SubNet
            self.sub_nets = nn.ModuleList([
                SubNet(
                    input_dim=input_dim,
                    tcn_hidden=tcn_hidden,
                    tcn_output=tcn_output,
                    lstm_hidden=lstm_hidden,
                    lstm_layers=lstm_layers,
                    unified_dim=unified_dim,
                    dropout=dropout,
                )
                for _ in range(n_imf)
            ])

    def _encode_externals(self, externals: torch.Tensor):
        """
        使用DAE编码器对原始外部特征进行编码

        同时缓存解码结果，供 compute_dae_loss() 使用，避免重复前向传播。

        Args:
            externals: 原始外部特征，形状 (batch, seq_len, n_external)

        Returns:
            tuple:
                - encoded: 编码后特征，形状 (batch, seq_len, dae_output_dim)
                - decoded: 解码重构特征，形状 (batch, seq_len, n_external)
        """
        batch_size, seq_len, n_raw = externals.shape
        # 展平时间维度: (batch*seq_len, n_raw)
        ext_flat = externals.reshape(-1, n_raw)
        # DAE编码+解码
        encoded_flat, decoded_flat = self.dae(ext_flat, noise=self.training)
        # 恢复时间维度
        encoded = encoded_flat.reshape(batch_size, seq_len, -1)
        decoded = decoded_flat.reshape(batch_size, seq_len, n_raw)
        # 缓存解码结果和原始输入，供 compute_dae_loss() 使用，避免重复前向传播
        self._cached_decoded = decoded
        self._cached_raw = externals
        self._cache_valid = True  # 标记缓存有效（当前 forward 调用产生）
        return encoded, decoded

    def compute_dae_loss(self, raw_externals: torch.Tensor) -> torch.Tensor:
        """
        计算DAE重构辅助损失

        在端到端训练模式下，重构损失作为正则项防止编码器崩塌。
        优先使用 forward() 中缓存的解码结果，避免对 DAE 进行重复前向传播
        （确保重构损失与实际编码使用的噪声实例一致，梯度更新正确）。

        使用 _cache_valid 标志位判断缓存是否来自当前 forward 调用，
        避免依赖 Python 对象同一性（is 检查），因为 PyTorch 切片操作
        每次创建新视图对象，导致 is 检查在 single 模型下始终为 False。

        Args:
            raw_externals: 原始外部特征，形状 (batch, seq_len, n_external)

        Returns:
            torch.Tensor: 重构MSE损失（标量）
        """
        if not self.use_dae:
            return torch.tensor(0.0, device=raw_externals.device)
        # 优先使用 forward() 中缓存的解码结果，避免重复前向传播
        # 注意：不能用 is 检查对象同一性，因为 PyTorch 切片操作每次创建新视图对象
        if getattr(self, '_cache_valid', False):
            # 验证缓存形状与当前输入一致
            if self._cached_decoded.shape == raw_externals.shape:
                self._cache_valid = False  # 消费缓存，防止重复使用
                return F.mse_loss(
                    self._cached_decoded.reshape(-1, self._cached_decoded.shape[-1]),
                    self._cached_raw.reshape(-1, self._cached_raw.shape[-1])
                )
        # 降级：如果缓存不可用（如独立调用），则重新计算
        batch_size, seq_len, n_raw = raw_externals.shape
        ext_flat = raw_externals.reshape(-1, n_raw)
        _, decoded_flat = self.dae(ext_flat, noise=self.training)
        return F.mse_loss(decoded_flat, ext_flat)

    def forward(self, imfs: torch.Tensor, externals: torch.Tensor):
        """
         前向传播过程

         处理流程:
         1. 如果use_dae=True，先将原始外部特征编码为低维表示
         2. 遍历每个IMF分量，提取对应的IMF数据
         3. 将当前IMF与（编码后的）外部特征在特征维度拼接
         4. 通过对应的SubNet子网络进行预测，获取预测值和注意力权重
         5. 累加所有IMF的预测值得到最终预测结果
         6. 收集所有子网络的注意力权重用于后续分析

         Args:
             imfs (torch.Tensor): IMF分量张量，形状为 (batch_size, seq_len, n_imf)
             externals (torch.Tensor): 外部特征张量，形状为 (batch_size, seq_len, n_external)

         Returns:
             tuple:
                 - total_pred (torch.Tensor): 最终预测值，形状为 (batch_size,)，为所有IMF预测值的累加和
                 - all_weights (list[torch.Tensor]): 注意力权重列表，长度为n_imf，
                   每个元素为对应SubNet的注意力权重，用于可视化各IMF的时间重要性分布
         """
        # DAE编码（端到端模式）
        if self.use_dae:
            encoded_externals, _ = self._encode_externals(externals)
            ext_for_subnet = encoded_externals
        else:
            ext_for_subnet = externals

        total_pred = 0.0
        all_weights = []

        # 遍历每个IMF分量，独立预测后累加
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