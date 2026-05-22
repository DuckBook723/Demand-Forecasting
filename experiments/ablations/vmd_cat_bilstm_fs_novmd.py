"""
文件位置: experiments/ablations/vmd_cat_bilstm_fs_novmd.py
作用: 消融实验 — 去除 VMD 分解 (w/o VMD)
      保留: FS + CATCN + BiLSTM + GateFusion + PredictionHead + Skip
      修改: 输入仅使用外部特征 + 原始需求（无 IMF 分量，无 delta_IMF）
      基础特征: 外部14 + delta_demand + demand = 16维
"""

import os
import sys
import random
import json
import time
import warnings

os.environ["PYTHONHASHSEED"] = str(42)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.models.trainer import calculate_metrics
from src.data.preprocess import (
    load_raw_data, clean_data_train, clean_data_with_stats, split_by_time
)
from configs.config import (
    DEMAND_COL, BATCH_SIZE, MODELS_DIR, DATA_FIGURES_DIR,
    SIM_DATASET_PATH, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
)

warnings.filterwarnings("ignore")


# ==================== 全局随机种子 ====================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.set_printoptions(precision=6, suppress=True)
    torch.set_printoptions(precision=6)
    print(f"🔒 全局随机种子已固定: seed={seed}")
    print(f" PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f" CUDA version: {torch.version.cuda}")
        print(f" GPU: {torch.cuda.get_device_name(0)}")


def _safe_float(val):
    if isinstance(val, np.ndarray):
        return float(val.item() if val.size == 1 else val.mean())
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FIGURES_DIR, exist_ok=True)


# ========================================================================
# Part 1: 多方法共识特征筛选器（无VMD版本：16维基础特征）
# ========================================================================
class ConsensusFeatureSelector:
    """
    【消融】去除VMD后，基础特征仅16维：
      外部14维 + delta_demand + demand
    """
    FEATURE_NAMES = [
        'temperature', 'extreme_weather_days', 'wind_speed',
        'macro_invest_growth', 'gdp_growth', 'copper_price_index',
        'power_load_MW', 'equipment_aging_rate', 'bidding_intensity',
        'renewable_capacity_MW', 'is_spring_festival',
        'month_sin', 'month_cos', 'quarter',
        'delta_demand', 'demand',
    ]

    def __init__(self, protected_names=None, corr_threshold=0.05,
                 redundancy_threshold=0.85, random_state=42):
        self.protected_names = protected_names or ['demand']
        self.corr_threshold = corr_threshold
        self.redundancy_threshold = redundancy_threshold
        self.random_state = random_state
        self.selected_indices_ = None
        self.selected_names_ = None
        self.method_scores_ = {}
        self.borda_scores_ = None
        self.removed_by_corr_ = []
        self.removed_by_redundancy_ = []
        self.n_original_ = None

    def fit(self, X, y, feature_names=None):
        if feature_names is None:
            feature_names = self.FEATURE_NAMES[:X.shape[1]]
        self.n_original_ = X.shape[1]
        n_features = X.shape[1]
        self.feature_names_ = feature_names
        self.protected_indices_ = set()
        for name in self.protected_names:
            if name in feature_names:
                self.protected_indices_.add(feature_names.index(name))

        print(f"\n{'=' * 70}")
        print("🔍 多方法共识特征筛选 (ConsensusFeatureSelector) — w/o VMD")
        print(f"{'=' * 70}")
        print(f" 输入特征数: {n_features}")
        print(f" 受保护特征: {self.protected_names}")
        print(f" SHAP可用: {'✅' if HAS_SHAP else '❌ (使用GBDT替代)'}")

        surviving = self._step1_correlation_filter(X, y, feature_names)
        self._step2_multi_method_ranking(X, y, surviving, feature_names)
        surviving = self._step3_redundancy_pruning(X, surviving, feature_names)
        self._step4_kneedle_cutoff(surviving, feature_names)

        for idx in self.protected_indices_:
            if idx not in self.selected_indices_:
                self.selected_indices_.append(idx)

        demand_idx = feature_names.index('demand') if 'demand' in feature_names else None
        if demand_idx is not None and demand_idx in self.selected_indices_:
            non_demand = [i for i in self.selected_indices_ if i != demand_idx]
            self.selected_indices_ = non_demand + [demand_idx]

        self.selected_names_ = [feature_names[i] for i in self.selected_indices_]
        self._print_summary(feature_names)
        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError("请先调用 fit()")
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, feature_names=None):
        self.fit(X, y, feature_names)
        return self.transform(X)

    def _step1_correlation_filter(self, X, y, feature_names):
        print(f"\n── Step 1: 相关性预筛选 (阈值={self.corr_threshold}) ──")
        n_features = X.shape[1]
        surviving = []
        for i in range(n_features):
            xi = X[:, i]
            try:
                sp_corr, _ = spearmanr(xi, y)
                sp_abs = abs(sp_corr) if not np.isnan(sp_corr) else 0.0
            except Exception:
                sp_abs = 0.0
            try:
                mi_val = mutual_info_regression(
                    xi.reshape(-1, 1), y,
                    n_neighbors=5, random_state=self.random_state
                )[0]
            except Exception:
                mi_val = 0.0
            mi_normalized = min(mi_val / (y.std() + 1e-8), 1.0)
            if sp_abs < self.corr_threshold and mi_normalized < self.corr_threshold:
                if i not in self.protected_indices_:
                    self.removed_by_corr_.append(i)
                    print(f"   ❌ 移除: {feature_names[i]:<<25} "
                          f"|ρ|={sp_abs:.4f}, MI_norm={mi_normalized:.4f}")
                    continue
            surviving.append(i)
        print(f"   结果: 保留 {len(surviving)}/{n_features} 个特征 "
              f"(移除 {len(self.removed_by_corr_)} 个无关特征)")
        return surviving

    def _step2_multi_method_ranking(self, X, y, surviving, feature_names):
        print(f"\n── Step 2: 多方法排序 + Borda计数投票 ──")
        X_sub = X[:, surviving]
        scores_dict = {}
        sp_scores = {}
        for i, idx in enumerate(surviving):
            try:
                corr, _ = spearmanr(X_sub[:, i], y)
                sp_scores[idx] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                sp_scores[idx] = 0.0
        scores_dict['Spearman'] = sp_scores

        mi_scores = {}
        try:
            mi_vals = mutual_info_regression(
                X_sub, y, n_neighbors=5, random_state=self.random_state)
            for i, idx in enumerate(surviving):
                mi_scores[idx] = mi_vals[i]
        except Exception:
            for idx in surviving:
                mi_scores[idx] = 0.0
        scores_dict['MI'] = mi_scores

        rf_scores = {}
        try:
            rf = RandomForestRegressor(
                n_estimators=200, max_depth=8,
                random_state=self.random_state, n_jobs=-1)
            rf.fit(X_sub, y)
            perm = permutation_importance(
                rf, X_sub, y, n_repeats=15,
                random_state=self.random_state, n_jobs=-1)
            for i, idx in enumerate(surviving):
                rf_scores[idx] = perm.importances_mean[i]
            min_val = min(rf_scores.values())
            if min_val < 0:
                rf_scores = {k: v - min_val for k, v in rf_scores.items()}
        except Exception:
            for idx in surviving:
                rf_scores[idx] = 0.0
        scores_dict['RF_Perm'] = rf_scores

        shap_scores = {}
        method4_name = "SHAP" if HAS_SHAP else "GBDT"
        try:
            if HAS_SHAP:
                gb = GradientBoostingRegressor(
                    n_estimators=150, max_depth=4,
                    learning_rate=0.1, random_state=self.random_state)
                gb.fit(X_sub, y)
                explainer = shap.TreeExplainer(gb)
                sv = explainer.shap_values(X_sub)
                shap_means = np.abs(sv).mean(axis=0)
                for i, idx in enumerate(surviving):
                    shap_scores[idx] = shap_means[i]
            else:
                gb = GradientBoostingRegressor(
                    n_estimators=150, max_depth=4,
                    learning_rate=0.1, random_state=self.random_state)
                gb.fit(X_sub, y)
                for i, idx in enumerate(surviving):
                    shap_scores[idx] = gb.feature_importances_[i]
        except Exception:
            for idx in surviving:
                shap_scores[idx] = 0.0
        scores_dict[method4_name] = shap_scores

        self.method_scores_ = scores_dict
        self.borda_scores_ = self._borda_count(scores_dict, surviving)
        self._print_ranking_table(scores_dict, surviving, feature_names)

    def _borda_count(self, scores_dict, surviving):
        n = len(surviving)
        borda = {idx: 0 for idx in surviving}
        for method_name, scores in scores_dict.items():
            ranked = sorted(surviving, key=lambda x: scores.get(x, 0), reverse=True)
            for rank, idx in enumerate(ranked):
                borda[idx] += (n - rank)
        return borda

    def _print_ranking_table(self, scores_dict, surviving, feature_names):
        sorted_indices = sorted(surviving,
                                 key=lambda x: self.borda_scores_[x], reverse=True)
        method_names = list(scores_dict.keys())
        header = f"{'特征名':<<25}"
        for mn in method_names:
            header += f" {mn:<10}"
        header += f" {'Borda':<<8}"
        print(f"   {header}")
        print(f"   {'─' * (25 + 11 * len(method_names) + 9)}")
        for idx in sorted_indices:
            row = f"   {feature_names[idx]:<<25}"
            for mn in method_names:
                val = scores_dict[mn].get(idx, 0)
                row += f" {val:<10.4f}"
            row += f" {self.borda_scores_[idx]:<<8.0f}"
            protected = "🔒" if idx in self.protected_indices_ else ""
            print(f"{row}{protected}")

    def _step3_redundancy_pruning(self, X, surviving, feature_names):
        print(f"\n── Step 3: 冗余剪枝 (阈值={self.redundancy_threshold}) ──")
        X_sub = X[:, surviving]
        n_sub = X_sub.shape[1]
        corr_matrix = np.zeros((n_sub, n_sub))
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                try:
                    c, _ = spearmanr(X_sub[:, i], X_sub[:, j])
                    corr_matrix[i, j] = abs(c) if not np.isnan(c) else 0.0
                    corr_matrix[j, i] = corr_matrix[i, j]
                except Exception:
                    corr_matrix[i, j] = 0.0
                    corr_matrix[j, i] = 0.0

        to_remove = set()
        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                if corr_matrix[i, j] > self.redundancy_threshold:
                    idx_i = surviving[i]
                    idx_j = surviving[j]
                    if idx_i in self.protected_indices_ or idx_j in self.protected_indices_:
                        continue
                    if idx_i in to_remove or idx_j in to_remove:
                        continue
                    score_i = self.borda_scores_.get(idx_i, 0)
                    score_j = self.borda_scores_.get(idx_j, 0)
                    if score_i >= score_j:
                        remove_idx = idx_j
                        keep_name = feature_names[idx_i]
                        remove_name = feature_names[idx_j]
                    else:
                        remove_idx = idx_i
                        keep_name = feature_names[idx_j]
                        remove_name = feature_names[idx_i]
                    to_remove.add(remove_idx)
                    self.removed_by_redundancy_.append(remove_idx)
                    print(f"   ✂️ 移除冗余: {remove_name:<20} "
                          f"(与 {keep_name} ρ={corr_matrix[i,j]:.3f}, "
                          f"Borda: {self.borda_scores_[remove_idx]:.0f} < {self.borda_scores_[surviving[i] if remove_idx == idx_j else surviving[j]]:.0f})")
        new_surviving = [idx for idx in surviving if idx not in to_remove]
        print(f"   结果: 保留 {len(new_surviving)}/{len(surviving)} 个特征 "
              f"(移除 {len(to_remove)} 个冗余特征)")
        return new_surviving

    def _step4_kneedle_cutoff(self, surviving, feature_names):
        print(f"\n── Step 4: Kneedle自适应截断 ──")
        sorted_by_borda = sorted(surviving,
                                  key=lambda x: self.borda_scores_[x], reverse=True)
        sorted_scores = np.array([self.borda_scores_[idx] for idx in sorted_by_borda],
                                  dtype=float)
        n = len(sorted_scores)
        if n <= 3:
            self.selected_indices_ = sorted_by_borda
            print(f"   特征数≤3, 全部保留")
            return
        x_norm = np.arange(n) / (n - 1)
        y_range = sorted_scores[0] - sorted_scores[-1]
        if y_range < 1e-10:
            self.selected_indices_ = sorted_by_borda
            print(f"   所有Borda分相近, 全部保留")
            return
        y_norm = (sorted_scores - sorted_scores[-1]) / y_range
        diff = y_norm - (1 - x_norm)
        if diff.max() <= 0:
            self.selected_indices_ = sorted_by_borda
            print(f"   无明显肘部, 全部保留")
            return
        knee_idx = np.argmax(diff)
        n_select = max(knee_idx + 1, 3)
        for idx in self.protected_indices_:
            if idx in sorted_by_borda and sorted_by_borda.index(idx) >= n_select:
                n_select = sorted_by_borda.index(idx) + 1
        self.selected_indices_ = sorted_by_borda[:n_select]
        print(f"   肘部位置: 第 {knee_idx + 1} 个特征 (累计Borda占比: "
              f"{sorted_scores[:n_select].sum() / sorted_scores.sum() * 100:.1f}%)")
        print(f"   选择: {n_select}/{n} 个特征")

    def _print_summary(self, feature_names):
        print(f"\n{'=' * 70}")
        print("✅ 特征筛选结果汇总")
        print(f"{'=' * 70}")
        print(f" 原始特征数:   {self.n_original_}")
        print(f" 相关性移除:   {len(self.removed_by_corr_)} 个 "
              f"{[feature_names[i] for i in self.removed_by_corr_]}")
        print(f" 冗余性移除:   {len(self.removed_by_redundancy_)} 个 "
              f"{[feature_names[i] for i in self.removed_by_redundancy_]}")
        print(f" 最终选择:     {len(self.selected_indices_)}/{self.n_original_} 个")
        print(f"\n 选中特征 (按Borda排序):")
        for i, idx in enumerate(self.selected_indices_):
            borda = self.borda_scores_.get(idx, 0)
            protected = " 🔒" if idx in self.protected_indices_ else ""
            print(f"   {i + 1:>2}. {feature_names[idx]:<<25} Borda={borda:.0f}{protected}")
        print(f"{'=' * 70}")

    def save(self, path):
        state = {
            "selected_indices": self.selected_indices_,
            "selected_names": self.selected_names_,
            "borda_scores": {str(k): v for k, v in self.borda_scores_.items()},
            "removed_by_corr": self.removed_by_corr_,
            "removed_by_redundancy": self.removed_by_redundancy_,
            "n_original": self.n_original_,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f" 💾 筛选器已保存: {path}")


# ========================================================================
# Part 2: 数据加载（无VMD：不使用IMF分解）
# ========================================================================
def load_vmd_catbilstm_data(seq_len=12, use_feature_selection=True):
    """
    【消融】去除VMD分解：
      - 不调用vmd_decompose
      - 不生成IMF1~4和delta_IMF
      - 基础特征: 外部14 + delta_demand + demand = 16维
    """
    print("📂 加载数据（w/o VMD：无 IMF 分解，仅外部特征+需求）...")
    df = load_raw_data(SIM_DATASET_PATH)
    train_raw, val_raw, test_raw = split_by_time(df, TRAIN_RATIO, VAL_RATIO)
    train_clean, clean_stats = clean_data_train(train_raw)
    val_clean = clean_data_with_stats(val_raw, clean_stats)
    test_clean = clean_data_with_stats(test_raw, clean_stats)

    train_demand = train_clean[DEMAND_COL].values
    val_demand = val_clean[DEMAND_COL].values
    test_demand = test_clean[DEMAND_COL].values

    # 外部特征（14维）
    external_cols = [
        'temperature', 'extreme_weather_days', 'wind_speed',
        'macro_invest_growth', 'gdp_growth', 'copper_price_index',
        'power_load_MW', 'equipment_aging_rate', 'bidding_intensity',
        'renewable_capacity_MW', 'is_spring_festival'
    ]
    for d in [train_clean, val_clean, test_clean]:
        d['month_sin'] = np.sin(2 * np.pi * d.index.month / 12)
        d['month_cos'] = np.cos(2 * np.pi * d.index.month / 12)
        d['quarter'] = d.index.quarter
    feature_cols = external_cols + ['month_sin', 'month_cos', 'quarter']

    # 【关键修改】不执行VMD，仅构建外部特征 + 动量 + 需求
    def build_features_no_vmd(demand_seg, X_base):
        delta_demand = np.zeros_like(demand_seg)
        delta_demand[1:] = demand_seg[1:] - demand_seg[:-1]
        # 拼接: 外部14 + delta_demand + demand = 16维
        X = np.column_stack([X_base, delta_demand, demand_seg])
        return X

    X_train_raw, X_val_raw, X_test_raw = None, None, None
    for idx, d in enumerate([train_clean, val_clean, test_clean]):
        X_base = d[feature_cols].values
        demand_seg = [train_demand, val_demand, test_demand][idx]
        X_seg = build_features_no_vmd(demand_seg, X_base)
        if idx == 0:
            X_train_raw = X_seg
        elif idx == 1:
            X_val_raw = X_seg
        else:
            X_test_raw = X_seg

    # 特征筛选（在16维上运行，与主模型流程一致）
    feature_selector = None
    selected_indices = None

    if use_feature_selection:
        feature_names = ConsensusFeatureSelector.FEATURE_NAMES[:X_train_raw.shape[1]]
        X_for_fs = X_train_raw[:-1]
        y_for_fs = train_demand[1:]

        feature_selector = ConsensusFeatureSelector(
            protected_names=['demand'],
            corr_threshold=0.05,
            redundancy_threshold=0.85,
            random_state=RANDOM_SEED,
        )
        feature_selector.fit(X_for_fs, y_for_fs, feature_names)
        selected_indices = feature_selector.selected_indices_

        fs_path = os.path.join(DATA_FIGURES_DIR, "feature_selector_state_novmd.json")
        feature_selector.save(fs_path)

        X_train_raw = X_train_raw[:, selected_indices]
        X_val_raw = X_val_raw[:, selected_indices]
        X_test_raw = X_test_raw[:, selected_indices]

        print(f"\n 特征筛选后维度: {X_train_raw.shape[-1]} 维")
    else:
        print(" ⏭️ 跳过特征筛选")

    # X 标准化（仅训练集 fit）
    scaler_X = StandardScaler()
    scaler_X.fit(X_train_raw)
    X_train_scaled = scaler_X.transform(X_train_raw)
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    # y 标准化（仅训练集统计量）
    y_mean = train_demand.mean()
    y_std = train_demand.std() + 1e-8

    X_scaled = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
    y_full = np.concatenate([train_demand, val_demand, test_demand])
    y_scaled = (y_full - y_mean) / y_std

    def create_seq(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_seq, y_seq = create_seq(X_scaled, y_scaled, seq_len)

    n_train = len(train_demand) - seq_len
    n_val = len(val_demand)

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]
    X_val = X_seq[n_train:n_train + n_val]
    y_val = y_seq[n_train:n_train + n_val]
    X_test = X_seq[n_train + n_val:]
    y_test = y_seq[n_train + n_val:]

    print(f" 输入特征维度: {X_train.shape[-1]} (外部14 + delta_demand + demand)")
    print(f" 训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test).reshape(-1, 1)),
        batch_size=BATCH_SIZE, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler_X": scaler_X,
        "y_mean": y_mean,
        "y_std": y_std,
        "seq_len": seq_len,
        "input_dim": X_train.shape[-1],
        "feature_selector": feature_selector,
        "selected_indices": selected_indices,
    }, train_demand, val_demand, test_demand


# ========================================================================
# Part 3: 模型架构（与主模型完全一致）
# ========================================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 dropout=0.3, eca_kernel_size=5):
        super().__init__()
        self.layer1 = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout),
        )
        self.net = nn.Sequential(self.layer1, self.layer2)
        self.eca = ECAAttention(eca_kernel_size)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        res = self.residual_conv(x)
        out = F.relu(out + res)
        return self.eca(out)


class CATCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=16,
                 num_layers=2, kernel_size=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layers = []
        for i in range(num_layers):
            layers.append(TemporalBlock(hidden_dim, hidden_dim, kernel_size,
                                        dilation=2 ** i, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x).transpose(1, 2)
        x = self.net(x).transpose(1, 2)
        return self.output_proj(x)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


class TABiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, bidirectional=True, dropout=0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.temporal_attention = TemporalAttention(hidden_dim * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.drop(lstm_out)
        context, weights = self.temporal_attention(lstm_out)
        return context, weights


class GateFusion(nn.Module):
    def __init__(self, local_dim=16, global_dim=64, unified_dim=32):
        super().__init__()
        self.local_proj = nn.Linear(local_dim, unified_dim)
        self.global_proj = nn.Linear(global_dim, unified_dim)
        self.gate = nn.Sequential(
            nn.Linear(unified_dim * 2, unified_dim), nn.ReLU(inplace=True),
            nn.Linear(unified_dim, unified_dim), nn.Sigmoid(),
        )

    def forward(self, local_feature, global_feature):
        if local_feature.dim() == 3:
            local_feature = local_feature.mean(dim=1)
        if global_feature.dim() == 3:
            global_feature = global_feature.mean(dim=1)
        lu = self.local_proj(local_feature)
        gu = self.global_proj(global_feature)
        gate = self.gate(torch.cat([lu, gu], dim=1))
        return gate * lu + (1 - gate) * gu


class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        layers = []
        cur = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(cur, h), nn.BatchNorm1d(h),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
            ])
            cur = h
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VMD_CAT_BiLSTM_NoVMD(nn.Module):
    """
    【消融】去除VMD分解，模型结构完全保留
    输入维度随特征筛选结果动态调整（约12~16维）
    """
    def __init__(self, input_dim, catcn_hidden=32, catcn_output=16,
                 catcn_layers=2, bilstm_hidden=32, bilstm_layers=1,
                 unified_dim=32, dropout=0.3):
        super().__init__()
        self.catcn = CATCN(input_dim, catcn_hidden, catcn_output,
                           num_layers=catcn_layers, dropout=dropout)
        self.tabilstm = TABiLSTM(input_dim, bilstm_hidden,
                                  num_layers=bilstm_layers, dropout=dropout)
        self.fusion = GateFusion(local_dim=catcn_output,
                                  global_dim=bilstm_hidden * 2,
                                  unified_dim=unified_dim)
        self.head = PredictionHead(unified_dim, hidden_dims=[32, 16],
                                    dropout=dropout)

    def forward(self, x):
        demand_last = x[:, -1, -1]  # 需求残差跳连保留
        f_local = self.catcn(x)
        f_global, attn_weights = self.tabilstm(x)
        fused = self.fusion(f_local, f_global)
        deep_delta = self.head(fused)
        out = deep_delta + 1.0 * demand_last
        return out


# ========================================================================
# Part 4: 训练（与主模型完全一致）
# ========================================================================
def train_model(model, data_dict, epochs=300, patience=30, lr=5e-4, device="cuda"):
    print(f"\n{'=' * 50}")
    print("🏋️ 开始训练（w/o VMD：无 IMF 分解）")
    print(f"{'=' * 50}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" 📦 参数量: {n_params:,}")
    print(f" 📋 lr={lr}, weight_decay=1e-3, dropout=0.3, 需求跳连=1.0固定")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).squeeze(-1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).squeeze(-1)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            ratio = val_loss / train_loss if train_loss > 0 else float('inf')
            cur_lr = optimizer.param_groups[0]['lr']
            y_mean, y_std = data_dict["y_mean"], data_dict["y_std"]
            vp = val_preds * y_std + y_mean
            vt = val_trues * y_std + y_mean
            vp = np.maximum(vp, 0)
            vm = calculate_metrics(vt, vp, is_normalized=False)
            print(f" Epoch {epoch}/{epochs} - "
                  f"Train:{train_loss:.5f}, Val:{val_loss:.5f}, V/T:{ratio:.1f}x, "
                  f"ValR²:{vm['R2']:.4f}, LR:{cur_lr:.2e}")

        if no_improve >= patience:
            print(f" ⏹️ 早停 epoch {epoch}")
            break

    model.load_state_dict(best_state)
    save_path = os.path.join(MODELS_DIR, "best_vmdcatbilstm_fs_novmd_seed42.pth")
    torch.save({
        "model_state": best_state,
        "input_dim": data_dict["input_dim"],
        "y_mean": float(data_dict["y_mean"]),
        "y_std": float(data_dict["y_std"]),
        "n_params": n_params,
    }, save_path, _use_new_zipfile_serialization=False)
    print(f" 💾 模型已保存: {save_path}")
    return model


# ========================================================================
# Part 5: 评估（与主模型完全一致）
# ========================================================================
def evaluate_model(model, data_dict, device="cuda"):
    model.eval()
    test_loader = data_dict["test_loader"]
    y_mean, y_std = data_dict["y_mean"], data_dict["y_std"]
    criterion = nn.MSELoss()

    total_loss = 0.0
    preds_list, trues_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            preds_list.append(pred.cpu().numpy())
            trues_list.append(yb.cpu().numpy())

    preds_z = np.concatenate(preds_list).flatten()
    trues_z = np.concatenate(trues_list).flatten()

    preds = preds_z * y_std + y_mean
    trues = trues_z * y_std + y_mean
    preds = np.maximum(preds, 0)

    metrics = calculate_metrics(trues, preds, is_normalized=False)
    return preds, trues, metrics


# ========================================================================
# Part 6: 实验主函数
# ========================================================================
def run_vmd_catbilstm_experiment(epochs=300, patience=30, lr=5e-4,
                                  device="cuda",
                                  use_feature_selection=True) -> pd.DataFrame:
    seed_everything(RANDOM_SEED)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ 使用设备: {device}")

    print("\n" + "=" * 70)
    print("🧪 VMD-CAT-BiLSTM-FS w/o VMD（无 IMF 分解）")
    print("=" * 70)

    t_start = time.time()
    data_dict, train_demand, val_demand, test_demand = load_vmd_catbilstm_data(
        seq_len=12, use_feature_selection=use_feature_selection)

    input_dim = data_dict["input_dim"]
    model = VMD_CAT_BiLSTM_NoVMD(input_dim=input_dim)

    model = train_model(model, data_dict, epochs, patience, lr, device)
    preds, trues, metrics = evaluate_model(model, data_dict, device)

    train_time_min = (time.time() - t_start) / 60.0

    print(f"\n{'=' * 50}")
    print("📊 w/o VMD 测试集结果")
    print(f"{'=' * 50}")
    print(f" RMSE: {metrics['RMSE']:.4f}")
    print(f" MAE:  {metrics['MAE']:.4f}")
    print(f" MAPE: {metrics['MAPE']:.2f}%")
    print(f" R²:   {metrics['R2']:.4f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" 参数量: {n_params:,}")
    print(f" 训练耗时: {train_time_min:.2f} min")

    print(f"\n 📈 对比主模型 (Full Model, R²=0.8858):")
    r2_diff = metrics['R2'] - 0.8858
    rmse_diff = metrics['RMSE'] - 303.26
    print(f"    R²  : {metrics['R2']:.4f} ({r2_diff:+.4f})")
    print(f"    RMSE: {metrics['RMSE']:.4f} ({rmse_diff:+.1f})")

    result_path = os.path.join(DATA_FIGURES_DIR, "vmd_catbilstm_novmd_predictions.csv")
    pd.DataFrame({
        "true_demand": trues.flatten(),
        "pred_demand": preds.flatten(),
    }).to_csv(result_path, index=False)

    results = {
        "name": "vmd_cat_bilstm_fs_novmd_seed42",
        "description": "Ablation: VMD-CAT-BiLSTM-FS w/o VMD (no IMF decomposition)",
        "RMSE": _safe_float(metrics.get("RMSE", -1)),
        "MAE": _safe_float(metrics.get("MAE", -1)),
        "MAPE": _safe_float(metrics.get("MAPE", -1)),
        "R2": _safe_float(metrics.get("R2", -1)),
        "n_params": n_params,
        "train_time_min": round(train_time_min, 2),
        "n_features": input_dim,
        "model_type": "vmd_cat_bilstm_fs_novmd",
    }
    results_df = pd.DataFrame([results])
    csv_path = os.path.join(DATA_FIGURES_DIR, "vmd_catbilstm_fs_novmd_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\n💾 结果已保存至: {csv_path}")

    log_path = os.path.join(DATA_FIGURES_DIR, "vmd_catbilstm_fs_novmd_log.json")
    log_data = {
        "seed": RANDOM_SEED,
        "pytorch_version": torch.__version__,
        "best_r2": float(results["R2"]),
        "best_rmse": float(results["RMSE"]),
        "n_params": n_params,
        "n_features": input_dim,
        "train_time_min": results["train_time_min"],
        "version": "ablation_novmd_no_imf",
    }
    if data_dict.get("selected_indices"):
        log_data["selected_indices"] = data_dict["selected_indices"]
        log_data["selected_names"] = data_dict["feature_selector"].selected_names_
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"📝 实验日志已保存: {log_path}")

    return results_df


# ========================================================================
# Part 7: 主入口
# ========================================================================
if __name__ == "__main__":
    seed_everything(seed=RANDOM_SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_FIGURES_DIR, exist_ok=True)

    results = run_vmd_catbilstm_experiment(
        epochs=300,
        patience=30,
        lr=5e-4,
        use_feature_selection=True,
    )