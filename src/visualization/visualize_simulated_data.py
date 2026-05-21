"""
文件位置: src/visualization/visualize_simulated_data.py
作用: 电网物资需求仿真数据全面可视化与质量诊断
      生成8张独立分析图 + 1张综合诊断面板，用于论文插图与数据质量校验

输出文件 (保存至 figures/ 目录):
    sim_data_01_demand_series.png      — 需求时间序列与趋势分解
    sim_data_02_seasonal_box.png       — 月度季节箱线图
    sim_data_03_acf_pacf.png           — ACF/PACF 自相关诊断
    sim_data_04_correlation_heatmap.png — 特征相关性热力图
    sim_data_05_demand_histogram.png   — 需求分布与正态性检验
    sim_data_06_scatter_matrix.png     — 需求-特征散点矩阵
    sim_data_07_feature_series.png     — 关键外部特征时序
    sim_data_08_zero_analysis.png      — 零需求月份统计
    sim_data_09_diagnosis_panel.png    — 综合诊断面板（4合1）

设计原则:
    - 配色遵循顶级期刊风格 (Nature/Science 冷色调为主，暖色仅用于强调)
    - 字体采用三层回退: 项目字体 → 系统中文 → matplotlib 默认
    - 所有图表支持 300 DPI 高清输出，适配论文印刷
    - 中文显示经过跨平台测试 (Windows/macOS/Linux)
"""

import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from configs.config import (
    SIM_DATASET_PATH,
    FIGURES_DIR,
    FONT_FILE_PATH,
)
warnings.filterwarnings("ignore")

# ==================== 学术配色方案 ====================
# 主色系: 深蓝-灰蓝-浅蓝 (冷色调，传递专业感)
# 强调色: 赭石橙 (仅用于高亮关键信息，避免视觉疲劳)
COLORS = {
    "primary": "#1a5276",      # 深海蓝 — 主曲线
    "secondary": "#5dade2",    # 天蓝 — 辅助线/置信带
    "accent": "#c0392b",       # 砖红 — 均值/异常高亮
    "success": "#27ae60",      # 翠绿 — 趋势/拟合
    "warning": "#d68910",      # 琥珀 — 警示信息
    "light": "#aed6f1",        # 淡蓝 — 背景填充
    "dark": "#2c3e50",         # 炭黑 — 文字
    "gray": "#7f8c8d",         # 灰 — 网格/次要元素
    "bg": "#f8f9fa",           # 近白 — 画布底色
    "grid": "#e9ecef",         # 浅灰 — 网格线
    "panel_bg": "#ffffff",     # 纯白 — 子图底色
}

# ==================== 字体配置引擎 ====================
def _configure_fonts(font_path: Optional[str]):
    """全局配置 Matplotlib 字体和样式"""
    # 尝试使用配置文件中的字体路径
    if font_path and os.path.exists(font_path):
        try:
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            print(f"✅ 成功加载指定字体: {font_path}")
            return font_prop.get_name()
        except Exception as e:
            print(f"⚠️ 加载指定字体失败: {e}，尝试回退到系统通用字体...")

    # 【修改点】将 SimHei (黑体) 放在第一位，它是 Windows/Linux 兼容性最好的中文字体
    candidate_system_fonts = [
        "SimHei",  # ⭐ 强制优先使用黑体
        "Microsoft YaHei",  # Windows 微软雅黑
        "PingFang SC",  # macOS 苹方
        "WenQuanYi Micro Hei",  # Linux 文泉驿微米黑
        "sans-serif"  # 最后的防线
    ]

    for font in candidate_system_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            # 测试是否能正常渲染
            plt.figure()
            plt.text(0.5, 0.5, '测试中文', fontproperties={'family': font})
            plt.close()
            print(f"✅ 成功设置系统字体: {font}")
            return font
        except Exception:
            continue

    print("❌ 未能找到可用的中文字体，图表中文可能显示为方框。")
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    return "sans-serif"


# 执行字体配置
ACTIVE_FONT = _configure_fonts(FONT_FILE_PATH)

# ==================== matplotlib 全局样式 ====================
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["panel_bg"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 18,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.labelpad": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": COLORS["dark"],
    "ytick.color": COLORS["dark"],
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "#cccccc",
    "legend.facecolor": "#ffffff",
    "legend.loc": "best",
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
})

sns.set_style("whitegrid")
sns.set_palette([
    COLORS["primary"], COLORS["secondary"], COLORS["accent"],
    COLORS["success"], COLORS["warning"], COLORS["gray"]
])

# 创建输出目录
os.makedirs(FIGURES_DIR, exist_ok=True)


# ======================== 1. 数据加载 ========================
def load_data() -> pd.DataFrame:
    """
    加载仿真数据集并解析日期索引。

    Returns:
        以 'date' 列为 DatetimeIndex 的 DataFrame，包含 demand 及外部特征列
    """
    df = pd.read_csv(SIM_DATASET_PATH, parse_dates=["date"], index_col="date")
    return df


# ======================== 2. 需求时间序列图 ========================
def plot_demand_series(df: pd.DataFrame) -> str:
    """
    绘制月度需求时间序列，叠加 12 个月移动平均趋势线与统计摘要文本框。

    Args:
        df: 包含 'demand' 列的时间序列 DataFrame

    Returns:
        保存的图像文件绝对路径
    """
    fig, ax = plt.subplots(figsize=(14, 5.5))

    # 主序列
    ax.plot(
        df.index, df["demand"],
        linewidth=1.2, color=COLORS["primary"], alpha=0.85,
        label="月度需求"
    )

    # 12 期移动平均
    rolling_mean = df["demand"].rolling(window=12, center=True, min_periods=1).mean()
    ax.plot(
        df.index, rolling_mean,
        linewidth=2.2, color=COLORS["accent"], linestyle="--",
        label="12 个月移动平均"
    )

    # 填充移动平均与真实值之间的偏差带（视觉降噪）
    ax.fill_between(
        df.index, df["demand"], rolling_mean,
        alpha=0.12, color=COLORS["secondary"], label="偏差带"
    )

    # 统计摘要文本框（右上角，避免遮挡曲线）
    stats_text = (
        f"样本量: {len(df)}\n"
        f"均值: {df['demand'].mean():.1f}\n"
        f"标准差: {df['demand'].std():.1f}\n"
        f"变异系数: {df['demand'].std()/df['demand'].mean():.2%}\n"
        f"最小值: {df['demand'].min():.0f}\n"
        f"最大值: {df['demand'].max():.0f}"
    )
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=COLORS["gray"], alpha=0.92)
    )

    ax.set_title("电网物资月度需求时间序列 (1995—2025)", fontsize=15, fontweight="bold")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("需求量", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_xlim(df.index.min(), df.index.max())

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "sim_data_01_demand_series.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 需求时间序列图已保存: {save_path}")
    return save_path


# ======================== 3. 月度季节箱线图 ========================
def plot_seasonal_box(df: pd.DataFrame) -> str:
    """
    绘制按月份分组的箱线图，展示需求的季节性分布特征。

    Args:
        df: 包含 'demand' 列的时间序列 DataFrame

    Returns:
        保存的图像文件绝对路径
    """
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly.index.month
    month_names = ["1月", "2月", "3月", "4月", "5月", "6月",
                   "7月", "8月", "9月", "10月", "11月", "12月"]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # 使用自定义调色板: 冬夏冷暖对比
    palette = sns.color_palette("RdYlBu_r", 12)
    sns.boxplot(
        x="month", y="demand", data=df_monthly,
        ax=ax, palette=palette, width=0.6,
        flierprops=dict(marker="o", markersize=3, alpha=0.4, color=COLORS["accent"])
    )

    ax.set_title("月度需求季节箱线图", fontsize=15, fontweight="bold")
    ax.set_xlabel("月份", fontsize=11)
    ax.set_ylabel("需求量", fontsize=11)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=10)

    # 添加月度均值连线，增强趋势可读性
    monthly_means = df_monthly.groupby("month")["demand"].mean()
    ax.plot(
        range(12), monthly_means.values,
        color=COLORS["accent"], marker="D", markersize=5,
        linewidth=1.5, linestyle="-", label="月度均值", zorder=5
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "sim_data_02_seasonal_box.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 季节箱线图已保存: {save_path}")
    return save_path


# ======================== 4. 自相关与偏自相关图 ========================
def plot_acf_pacf(df: pd.DataFrame, lags: int = 40) -> str:
    """
    绘制 ACF 与 PACF 图，用于判断时间序列的平稳性与季节性阶数。

    Args:
        df: 包含 'demand' 列的时间序列 DataFrame
        lags: 最大滞后阶数，默认 40（覆盖 3 年以上月度数据）

    Returns:
        保存的图像文件绝对路径
    """
    # 【修复维度不匹配】使用插值填充缺失值
    demand = df["demand"].interpolate(method='linear', limit_direction='both')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(demand, lags=lags, ax=axes[0], alpha=0.05, color=COLORS["primary"])
    axes[0].set_title("自相关函数 (ACF)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("滞后阶数", fontsize=10)
    axes[0].set_ylabel("自相关系数", fontsize=10)

    plot_pacf(demand, lags=lags, ax=axes[1], alpha=0.05, method="ywm", color=COLORS["primary"])
    axes[1].set_title("偏自相关函数 (PACF)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("滞后阶数", fontsize=10)
    axes[1].set_ylabel("偏自相关系数", fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(y=0, color=COLORS["gray"], linewidth=0.8)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "sim_data_03_acf_pacf.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 自相关图已保存: {save_path}")
    return save_path


# ======================== 5. 特征相关性热力图 ========================
def plot_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.08) -> str:
    """
    绘制特征与 demand 的相关性热力图，自动筛选与需求相关性高于阈值的特征。

    Args:
        df: 包含 demand 及外部特征的 DataFrame
        threshold: 相关性绝对值筛选阈值，低于该值的特征被排除以减少视觉噪声

    Returns:
        保存的图像文件绝对路径
    """
    corr_with_demand = df.corr()["demand"].abs().sort_values(ascending=False)
    top_features = corr_with_demand[corr_with_demand > threshold].index.tolist()

    if len(top_features) <= 2:
        top_features = corr_with_demand.head(10).index.tolist()

    if "demand" not in top_features:
        top_features.insert(0, "demand")

    df_sub = df[top_features]
    corr_matrix = df_sub.corr()

    n_features = len(top_features)
    figsize = min(14, max(8, n_features * 1.2))
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.6,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        annot_kws={"size": 8 if n_features > 12 else 9},
        ax=ax,
        vmin=-1, vmax=1,
    )

    ax.set_title("特征相关性热力图", fontsize=15, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "sim_data_04_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 相关性热力图已保存: {save_path}")
    return save_path


# ======================== 6. 需求分布直方图 ========================
def plot_demand_histogram(df: pd.DataFrame) -> str:
    """
    绘制需求分布直方图，叠加核密度估计 (KDE)、均值/中位数参考线，
    并执行 Shapiro-Wilk 正态性检验（若样本量允许）。

    Args:
        df: 包含 'demand' 列的 DataFrame

    Returns:
        保存的图像文件绝对路径
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # 直方图 + KDE
    sns.histplot(
        df["demand"], bins=30, kde=True, ax=ax,
        color=COLORS["primary"], edgecolor="white", alpha=0.85, stat="density"
    )

    # 参考线
    mean_val = df['demand'].mean()
    median_val = df['demand'].median()
    ax.axvline(
        mean_val, color=COLORS["accent"], linestyle="--", linewidth=2,
        label=f"均值: {mean_val:.1f}"
    )
    ax.axvline(
        median_val, color=COLORS["success"], linestyle="-.", linewidth=2,
        label=f"中位数: {median_val:.1f}"
    )

    # 正态性检验文本
    if len(df) <= 5000:
        stat, pvalue = stats.shapiro(df["demand"].sample(min(5000, len(df))))
        norm_text = f"Shapiro-Wilk 检验: W={stat:.3f}, p={pvalue:.3f}"
        if pvalue < 0.05:
            norm_text += "\n拒绝正态性假设 (p<<0.05)"
        else:
            norm_text += "\n无法拒绝正态性假设"
    else:
        stat, pvalue = stats.normaltest(df["demand"])
        norm_text = f"D'Agostino 检验: χ²={stat:.2f}, p={pvalue:.3f}"

    ax.text(
        0.97, 0.97, norm_text,
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=COLORS["gray"], alpha=0.92)
    )

    ax.set_title("需求分布直方图与正态性诊断", fontsize=15, fontweight="bold")
    ax.set_xlabel("需求量", fontsize=11)
    ax.set_ylabel("概率密度", fontsize=11)
    ax.legend(loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "sim_data_05_demand_histogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 需求直方图已保存: {save_path}")
    return save_path


# ======================== 7. 散点图矩阵 ========================
def plot_scatter_matrix(df: pd.DataFrame, top_n: int = 5) -> Optional[str]:
    """
    绘制需求与 top_n 个最相关特征的散点图矩阵（对角线为直方图）。

    Args:
        df: 包含 demand 及外部特征的 DataFrame
        top_n: 选取与 demand 相关性最高的特征数量，默认 5

    Returns:
        保存的图像文件绝对路径；若可用特征不足则返回 None
    """
    corr = df.corr()["demand"].abs().sort_values(ascending=False)
    top_features = corr.index[1 : top_n + 1].tolist()

    if not top_features:
        print("⚠️ 没有足够特征，跳过散点图矩阵")
        return None

    selected_cols = ["demand"] + top_features
    df_selected = df[selected_cols].dropna() # 依然需要dropna来确保每个子图的x,y对齐

    n = len(selected_cols)
    fig, axes = plt.subplots(n, n, figsize=(max(10, n * 2.2), max(10, n * 2.2)))

    # 统一坐标范围，增强可比性
    global_min = df_selected.min().min()
    global_max = df_selected.max().max()

    for i, row_var in enumerate(selected_cols):
        for j, col_var in enumerate(selected_cols):
            ax = axes[i, j]

            if i == j:
                # 对角线: 直方图 + KDE
                ax.hist(
                    df_selected[col_var], bins=18,
                    color=COLORS["secondary"], edgecolor="white", alpha=0.8
                )
                ax.set_title(col_var, fontsize=10, fontweight="bold", pad=6)
            else:
                # 非对角线: 散点图 + 回归线
                sns.regplot(
                    x=df_selected[col_var], y=df_selected[row_var],
                    ax=ax, scatter_kws={"s": 15, "alpha": 0.6, "color": COLORS["primary"]}, # 【修复散点图空白】
                    line_kws={"color": COLORS["accent"], "linewidth": 1.5},
                    ci=None, truncate=True
                )
                # 显示相关系数
                r = df_selected[col_var].corr(df_selected[row_var])
                ax.annotate(
                    f"r={r:.2f}",
                    xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=8, color=COLORS["dark"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                )

            # 统一刻度
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)

            # 标签控制：仅最左列和最下行显示轴标签
            if i == n - 1:
                ax.set_xlabel(col_var, fontsize=9)
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(row_var, fontsize=9)
            else:
                ax.set_ylabel("")

            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

    plt.suptitle("需求与主要特征散点图矩阵", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "sim_data_06_scatter_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 散点图矩阵已保存: {save_path}")
    return save_path


# ======================== 8. 关键特征时间序列 ========================
def plot_feature_time_series(
    df: pd.DataFrame, features: Optional[List[str]] = None, max_features: int = 5
) -> Optional[str]:
    """
    绘制与 demand 相关性最高的外部特征时间序列子图。

    Args:
        df: 包含 demand 及外部特征的 DataFrame
        features: 手动指定的特征列表；若为 None 则自动按相关性选取
        max_features: 自动选取时的特征数量上限

    Returns:
        保存的图像文件绝对路径；若无可选特征则返回 None
    """
    if features is None:
        corr = df.corr()["demand"].abs().sort_values(ascending=False)
        features = [c for c in corr.index[1 : max_features + 1] if c in df.columns]

    if not features:
        print("⚠️ 没有可用的特征，跳过特征时间序列图")
        return None

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, features):
        ax.plot(
            df.index, df[col],
            linewidth=1.3, color=COLORS["secondary"], alpha=0.9
        )
        ax.fill_between(df.index, df[col], alpha=0.15, color=COLORS["light"])
        ax.set_ylabel(col, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 添加滚动均值
        roll = df[col].rolling(window=12, center=True, min_periods=1).mean()
        ax.plot(
            df.index, roll,
            linewidth=1.5, color=COLORS["accent"], linestyle="--", alpha=0.8,
            label="12期移动平均"
        )
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("日期", fontsize=11)
    plt.suptitle("关键外部特征时间序列", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "sim_data_07_feature_series.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 特征时间序列图已保存: {save_path}")
    return save_path


# ======================== 9. 零需求月份分析 ========================
def plot_zero_demand_analysis(df: pd.DataFrame) -> Optional[str]:
    """
    统计并可视化零需求月份的年度分布与季节分布。

    Args:
        df: 包含 'demand' 列的时间序列 DataFrame

    Returns:
        保存的图像文件绝对路径；若无零需求月份则返回 None
    """
    zero_months = df[df["demand"] == 0]
    if len(zero_months) == 0:
        print("ℹ️ 未检测到零需求月份，跳过零需求分析图")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 各年份零需求月份数
    zero_by_year = zero_months.groupby(zero_months.index.year).size()
    years = zero_by_year.index
    axes[0].bar(
        years, zero_by_year.values,
        color=COLORS["warning"], edgecolor="white", width=0.7, alpha=0.85
    )
    axes[0].set_title("各年份零需求月份数", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("年份", fontsize=11)
    axes[0].set_ylabel("零需求月份数", fontsize=11)
    axes[0].set_xticks(years)
    axes[0].set_xticklabels(years, rotation=45, ha="right", fontsize=8)

    # 右图: 零需求的季节分布
    zero_by_month = zero_months.groupby(zero_months.index.month).size()
    month_names = ["1月", "2月", "3月", "4月", "5月", "6月",
                   "7月", "8月", "9月", "10月", "11月", "12月"]
    counts = zero_by_month.reindex(range(1, 13), fill_value=0).values

    bars = axes[1].bar(
        month_names, counts,
        color=[COLORS["accent"] if c > 0 else COLORS["gray"] for c in counts],
        edgecolor="white", width=0.6, alpha=0.85
    )
    axes[1].set_title("零需求月份的季节分布", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("月份", fontsize=11)
    axes[1].set_ylabel("零需求次数", fontsize=11)

    # 标注数值
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[1].annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "sim_data_08_zero_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 零需求分析图已保存: {save_path}")
    return save_path


# ======================== 10. 综合诊断面板 ========================
def plot_diagnosis_panel(df: pd.DataFrame) -> str:
    """
    生成 2×2 综合诊断面板，整合时间序列、ACF、分布和季节特征，
    用于论文中的"数据概览"单图展示。

    Args:
        df: 包含 'demand' 列的时间序列 DataFrame

    Returns:
        保存的图像文件绝对路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS["bg"])

    # 【修复维度不匹配】使用插值填充缺失值
    demand = df["demand"].interpolate(method='linear', limit_direction='both')

    # --- Panel A: 时间序列 + 趋势 ---
    ax = axes[0, 0]
    ax.plot(df.index, demand, linewidth=1.0, color=COLORS["primary"], alpha=0.7, label="原始序列")
    trend = demand.rolling(window=12, center=True, min_periods=1).mean()
    ax.plot(df.index, trend, linewidth=2.5, color=COLORS["accent"], label="12期趋势")
    ax.fill_between(df.index, demand, trend, alpha=0.1, color=COLORS["secondary"])
    ax.set_title("(a) 需求时间序列与趋势", fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel("需求量")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(df.index.min(), df.index.max())

    # --- Panel B: ACF (前 24 阶) ---
    ax = axes[0, 1]
    from statsmodels.tsa.stattools import acf
    acf_vals, confint = acf(demand, nlags=24, alpha=0.05, fft=True)
    lags = np.arange(len(acf_vals))
    ax.bar(lags, acf_vals, color=COLORS["primary"], width=0.6, edgecolor="white")
    ax.axhline(y=0, color=COLORS["dark"], linewidth=0.8)
    # 置信带
    lower = confint[:, 0] - acf_vals
    upper = confint[:, 1] - acf_vals
    ax.fill_between(lags, lower, upper, alpha=0.25, color=COLORS["secondary"], label="95% 置信区间")
    ax.set_title("(b) 自相关函数 (ACF)", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("滞后阶数")
    ax.set_ylabel("自相关系数")
    ax.set_xlim(-0.5, 24.5)
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel C: 分布直方图 + 正态拟合 ---
    ax = axes[1, 0]
    sns.histplot(demand, bins=25, kde=False, ax=ax, color=COLORS["secondary"], alpha=0.7, stat="density")
    # 叠加正态拟合曲线
    mu, sigma = demand.mean(), demand.std()
    x_norm = np.linspace(demand.min(), demand.max(), 200)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    ax.plot(x_norm, y_norm, color=COLORS["accent"], linewidth=2, linestyle="--", label=f"N({mu:.0f}, {sigma:.0f}²)")
    ax.axvline(mu, color=COLORS["warning"], linestyle="-", linewidth=1.5, label=f"均值={mu:.0f}")
    ax.set_title("(c) 需求分布与正态拟合", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("需求量")
    ax.set_ylabel("概率密度")
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel D: 季节小提琴图 ---
    ax = axes[1, 1]
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly.index.month
    month_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    sns.violinplot(
        x="month", y="demand", data=df_monthly, ax=ax,
        palette="RdYlBu_r", inner="box", linewidth=0.8
    )
    ax.set_title("(d) 月度需求分布 (小提琴图)", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("月份")
    ax.set_ylabel("需求量")
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=9)

    # 统一美化
    for ax in axes.flat:
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_facecolor(COLORS["panel_bg"])

    plt.suptitle(
        "电网物资仿真数据综合诊断面板",
        fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    save_path = os.path.join(FIGURES_DIR, "sim_data_09_diagnosis_panel.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 综合诊断面板已保存: {save_path}")
    return save_path


# ======================== 主函数 ========================
def main() -> None:
    """
    可视化主入口：顺序执行所有诊断图表生成函数，
    并在控制台输出数据质量摘要。
    """
    print("=" * 60)
    print("📊 电网物资仿真数据可视化诊断")
    print("=" * 60)

    df = load_data()
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df.index.min().date()} 至 {df.index.max().date()}")
    print(f"活跃字体: {ACTIVE_FONT}")

    # 快速质量摘要
    demand = df["demand"]
    print(f"\n📋 数据质量摘要:")
    print(f"   样本量: {len(demand)}")
    print(f"   缺失值: {demand.isna().sum()} ({demand.isna().mean():.2%})")
    print(f"   零需求月: {(demand == 0).sum()} ({(demand == 0).mean():.2%})")
    print(f"   均值: {demand.mean():.2f} | 标准差: {demand.std():.2f}")
    print(f"   偏度: {demand.skew():.3f} | 峰度: {demand.kurtosis():.3f}")
    acf_1 = demand.autocorr(lag=1)
    acf_12 = demand.autocorr(lag=12)
    print(f"   lag-1 自相关: {acf_1:.3f} | lag-12 自相关: {acf_12:.3f}")

    print("\n🎨 生成可视化图表...")
    plot_demand_series(df)
    plot_seasonal_box(df)
    plot_acf_pacf(df)
    plot_correlation_heatmap(df)
    plot_demand_histogram(df)
    plot_scatter_matrix(df, top_n=5)
    plot_feature_time_series(df)
    plot_zero_demand_analysis(df)
    plot_diagnosis_panel(df)

    print("\n" + "=" * 60)
    print("✅ 所有可视化图表生成完毕！")
    print(f"📁 输出目录: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
