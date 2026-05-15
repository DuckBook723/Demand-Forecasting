import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from src.models.ca_tcn import CausalConv1d
from configs.config import FIGURES_DIR

# 全局中文字体与样式配置
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 学术配色方案
COLORS = {
    'primary': '#2E5090',      # 深蓝
    'secondary': '#4A90A4',    # 青蓝
    'accent': '#D9534F',       # 强调红
    'highlight': '#F0AD4E',    # 暖橙
    'light': '#A8D0E6',        # 浅蓝
    'text': '#333333',         # 深灰文字
    'grid': '#E0E0E0',         # 网格灰
    'bg': '#FAFBFC'            # 背景色
}




class CausalConvVisualizer:
    """基于 CausalConv1d 模块的因果空洞卷积可视化工具"""

    def __init__(self, kernel_size=3, dilation=1):
        self.kernel_size = kernel_size
        self.dilation = dilation

    def create_causal_mask(self, input_length=8, dilation=None):
        """创建因果掩码矩阵，严格对齐 CausalConv1d 的逻辑"""
        d = dilation if dilation is not None else self.dilation
        mask = np.zeros((input_length, input_length))
        for t in range(input_length):
            for k in range(self.kernel_size):
                idx = t - k * d
                if idx >= 0:
                    mask[t, idx] = 1
        return mask

    def plot_causal_conv_process(self, save_path=None):
        """绘制因果空洞卷积全过程示意图"""
        if save_path is None:
            save_path = f'{FIGURES_DIR}/causal_dilated_conv_mechanism.png'

        fig = plt.figure(figsize=(16, 11), facecolor=COLORS['bg'])
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                               left=0.08, right=0.95, top=0.90, bottom=0.08)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])
        ]

        input_length = 8
        t = np.arange(input_length)
        np.random.seed(42)
        input_signal = np.sin(0.5 * t) + 0.3 * np.random.randn(input_length)

        # ========== 子图1：输入时间序列 ==========
        ax = axes[0]
        ax.set_facecolor(COLORS['bg'])
        ax.plot(t, input_signal, 'o-', linewidth=2.5, markersize=9,
                color=COLORS['primary'], markerfacecolor='white',
                markeredgewidth=2, zorder=3)
        ax.fill_between(t, input_signal, alpha=0.25, color=COLORS['light'])
        ax.set_title('(a) 输入时间序列 X(t)', fontsize=13, fontweight='bold',
                     color=COLORS['text'], pad=12)
        ax.set_xlabel('时间步 t', fontsize=11)
        ax.set_ylabel('信号幅值', fontsize=11)
        ax.set_xticks(t)
        ax.grid(True, alpha=0.4, color=COLORS['grid'], linestyle='--')
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
            spine.set_linewidth(0.8)

        # ========== 子图2：因果空洞卷积掩码热力图 ==========
        ax = axes[1]
        ax.set_facecolor(COLORS['bg'])
        mask = self.create_causal_mask(input_length)
        sns.heatmap(mask, ax=ax, cmap='Blues', annot=True, fmt='.0f',
                    annot_kws={'size': 10, 'weight': 'bold'},
                    cbar_kws={'label': '连接权重', 'shrink': 0.8},
                    square=True, linewidths=0.5, linecolor='white')
        ax.set_title('(b) 因果空洞卷积掩码矩阵', fontsize=13, fontweight='bold',
                     color=COLORS['text'], pad=12)
        ax.set_xlabel('输入时间 t', fontsize=11)
        ax.set_ylabel('输出时间 t', fontsize=11)
        # 因果边界线
        ax.plot([0, input_length], [0, input_length], 'r--',
                alpha=0.7, linewidth=2, label='因果边界（禁止窥视未来）')
        ax.legend(loc='upper left', framealpha=0.9, edgecolor='#CCCCCC')

        # ========== 子图3：卷积核滑动过程 ==========
        ax = axes[2]
        ax.set_facecolor(COLORS['bg'])
        input_tensor = torch.FloatTensor(input_signal).view(1, 1, -1)
        conv_layer = CausalConv1d(
            in_channels=1, out_channels=1,
            kernel_size=self.kernel_size, dilation=self.dilation
        )
        with torch.no_grad():
            conv_layer.conv.weight.data = torch.tensor([[[0.2, 0.5, 0.3]]])
            conv_layer.conv.bias.data = torch.zeros(1)
        output_tensor = conv_layer(input_tensor)
        output_signal = output_tensor.detach().numpy().flatten()

        ax.plot(t, output_signal, 's-', linewidth=2.5, markersize=9,
                color=COLORS['accent'], markerfacecolor='white',
                markeredgewidth=2, zorder=3)
        ax.set_title(
            f'(c) 因果卷积输出（长度保持：{len(output_signal)}）',
            fontsize=13, fontweight='bold', color=COLORS['text'], pad=12
        )
        ax.set_xlabel('时间步 t', fontsize=11)
        ax.set_ylabel('输出幅值', fontsize=11)
        ax.set_xticks(t)
        ax.grid(True, alpha=0.4, color=COLORS['grid'], linestyle='--')
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
            spine.set_linewidth(0.8)

        # ========== 子图4：感受野示意图 ==========
        ax = axes[3]
        ax.set_facecolor(COLORS['bg'])
        receptive_field = (self.kernel_size - 1) * self.dilation + 1
        current_t = input_length - 1
        start_t = current_t - receptive_field + 1

        # 绘制时间轴基线
        ax.plot(t, np.zeros_like(t), '-', color='#BBBBBB', linewidth=1.5, zorder=1)
        ax.plot(t, np.zeros_like(t), 'o', color=COLORS['secondary'],
                markersize=10, zorder=2)

        # 感受野覆盖区域
        rect = FancyBboxPatch(
            (start_t - 0.45, -0.18), receptive_field - 0.1, 0.36,
            boxstyle="round,pad=0.02", linewidth=2.5,
            edgecolor=COLORS['highlight'], facecolor=COLORS['highlight'],
            alpha=0.25, linestyle='--', zorder=1
        )
        ax.add_patch(rect)

        # 当前时刻高亮
        ax.plot(current_t, 0, 'o', markersize=18, markerfacecolor='none',
                markeredgecolor=COLORS['accent'], markeredgewidth=2.5, zorder=4)
        ax.annotate('当前时刻', xy=(current_t, 0), xytext=(current_t, 0.28),
                    fontsize=11, ha='center', color=COLORS['accent'],
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent'],
                                    lw=1.5))

        # 感受野标注
        ax.annotate(
            f'感受野窗口（长度={receptive_field}）',
            xy=((start_t + current_t) / 2, -0.12),
            fontsize=11, ha='center', color=COLORS['highlight'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['highlight'], alpha=0.9)
        )

        ax.set_xlim(-0.8, input_length + 0.5)
        ax.set_ylim(-0.45, 0.45)
        ax.set_title('(d) 因果感受野（仅回溯历史）', fontsize=13, fontweight='bold',
                     color=COLORS['text'], pad=12)
        ax.set_xlabel('时间步 t', fontsize=11)
        ax.axis('off')

        # 总标题
        fig.suptitle(
            f'因果空洞卷积机制原理图（卷积核={self.kernel_size}，膨胀系数={self.dilation}）',
            fontsize=15, fontweight='bold', color=COLORS['text'], y=0.96
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor=COLORS['bg'], edgecolor='none')
        print(f"✅ 因果空洞卷积机制图已保存至: {save_path}")
        plt.close()

    def plot_dilation_comparison(self, save_path=None):
        """对比不同 dilation 因子的感受野"""
        if save_path is None:
            save_path = f'{FIGURES_DIR}/dilation_comparison.png'

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=COLORS['bg'])
        plt.subplots_adjust(wspace=0.25, left=0.06, right=0.94, top=0.82, bottom=0.15)
        dilations = [1, 2, 4]

        for ax, d in zip(axes, dilations):
            ax.set_facecolor(COLORS['bg'])
            mask = self.create_causal_mask(dilation=d)
            sns.heatmap(mask, ax=ax, cmap='Greens', annot=True, fmt='.0f',
                        annot_kws={'size': 11, 'weight': 'bold'},
                        cbar=False, square=True,
                        linewidths=0.5, linecolor='white')

            rf_size = 1 + (self.kernel_size - 1) * d
            ax.set_title(
                f'膨胀系数 d = {d}\n感受野大小 = {rf_size}',
                fontsize=13, fontweight='bold', color=COLORS['text'], pad=10
            )
            ax.set_xlabel('输入时间 t', fontsize=11)
            ax.set_ylabel('输出时间 t', fontsize=11)

            # 未来信息屏蔽标注
            ax.text(0.5, 0.02, '⚠ 未来信息已屏蔽（零填充）',
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=10, color=COLORS['accent'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0',
                              edgecolor=COLORS['accent'], alpha=0.9))

        fig.suptitle(
            '膨胀系数对感受野的影响\n（不增加参数的前提下扩展感知范围）',
            fontsize=15, fontweight='bold', color=COLORS['text'], y=0.98
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor=COLORS['bg'], edgecolor='none')
        print(f"✅ Dilation 对比图已保存至: {save_path}")
        plt.close()


# 使用示例
if __name__ == '__main__':
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)

    visualizer = CausalConvVisualizer(kernel_size=3, dilation=1)
    visualizer.plot_causal_conv_process()
    visualizer.plot_dilation_comparison()