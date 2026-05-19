"""
电网物资需求仿真数据生成引擎 (v2.1 — 修复因果强度与零需求比例)

文件位置: src/data/generate_data.py
核心作用:
    本模块是整个需求预测项目的数据起点。采用"日度仿真→月度聚合"的两阶段策略：
    - 阶段一（日度仿真）：按业务逻辑模拟20种物资在5个区域的每日需求，
      包含三种需求模式（项目脉冲/连续消耗/零星随机），以及天气、节假日、
      BOM联动、突发事件等多重外部因素调制。
    - 阶段二（月度聚合）：将日度需求汇总为月度总量，并独立生成月度外部特征，
      输出与现有流水线完全兼容的月度数据表。

设计理念:
    日度仿真保证业务真实性——零需求、节假日效应、项目脉冲在日度才能真实模拟；
    月度聚合适配模型架构——VMD分解、CA-TCN、TA-BiLSTM都需要足够长的平滑序列。

输出文件:
    - grid_material_demand_sim.csv : 汇总月度需求 + 11个外部特征（主输出，兼容现有流水线）
    - material_monthly/            : 各物资-区域的独立月度需求序列（供P3跨物资实验使用）
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from configs.config import (
    SIM_DATASET_PATH,
    DATA_RAW_DIR,
    MATERIAL_DEMAND_PATH
)

# ==================== 全局参数配置 ====================
RANDOM_SEED = 42

NUM_MATERIALS = 20                      # 物资数量
REGIONS = ["A", "B", "C", "D", "E"]     # 区域列表
PROJECTS_PER_YEAR = 25                  # 每年项目数量

# 🔧 修改点：因果敏感度系数增强（提升宏观/招标/温度与需求的关联性）
# 原值: macro=10, bidding=1.5, temp=0.12 → 新值: macro=25, bidding=4.0, temp=0.35
CAUSAL_SENSITIVITY = {
    'macro_invest_growth': 25.0,
    'bidding_intensity': 4.0,
    'equipment_aging_rate': 8.0,
    'temperature': 0.35,
    'gdp_growth': 15.0,
    'copper_price_index': -1/200000,  # 负相关
    'power_load_MW': 1/25000,
    'extreme_weather_days': 0.3,
}

# ---------- 区域需求乘数（模拟区域经济/网架规模差异）----------
_REGION_SEED_STATE = np.random.get_state()
np.random.seed(123)
REGION_DEMAND_FACTOR = {r: round(np.random.uniform(0.8, 1.2), 2) for r in REGIONS}
np.random.set_state(_REGION_SEED_STATE)

# ---------- 差异化自回归系数 ----------
AR_COEFF_CONTINUOUS = 0.6               # 连续型：强惯性，需求曲线平滑
AR_COEFF_PROJECT = 0.3                  # 项目型：中等惯性
AR_COEFF_SPORADIC = 0.05                # 零星型：极弱惯性，消除虚假拖尾

# ---------- 外部因素系数 ----------
WORKDAY_BOOST = 1.0
PROJECT_WEEKEND_FACTOR = 0.75           # 项目型周末衰减
PROJECT_HOLIDAY_FACTOR = 0.60           # 项目型节假日衰减
EXTREME_HOT_FACTOR = 3.0                # 极端高温连续需求激增
EXTREME_COLD_FACTOR = 2.2               # 极端低温连续需求激增
COMFORT_TEMP = 20.0                     # 舒适温度基线

# ---------- BOM联动 ----------
MATERIAL_CORRELATION = {
    "MAT002": ["MAT003", "MAT011", "MAT012"],   # 钢芯铝绞线→绝缘子、金具
    "MAT001": ["MAT005", "MAT016"],              # 变压器→电缆分支箱、保护装置
}
CORRELATION_PROB = 0.50
CORRELATION_RATIO_MIN = 0.10
CORRELATION_RATIO_MAX = 0.30

# ---------- 突发事件冲击 ----------
SHOCK_PROBABILITY = 0.005
SHOCK_MULTIPLIER = 8.0
SHOCK_DURATION = 5
SHOCK_CATEGORIES = {"塔材", "导线", "绝缘子", "金具"}


# ======================== 1. 模拟天气 ========================
def simulate_daily_weather(date: datetime) -> Tuple[float, int]:
    """根据日期模拟日平均气温及是否极端天气。"""
    month = date.month
    if month in [12, 1, 2]:
        base_temp = 0.0
    elif month in [3, 4, 5]:
        base_temp = 12.0
    elif month in [6, 7, 8]:
        base_temp = 28.0
    else:
        base_temp = 18.0
    temp = base_temp + np.random.normal(0, 5)
    is_extreme = 1 if temp > 35 or temp < -10 else 0
    return round(temp, 1), is_extreme


# ======================== 2. 简易节假日判断 ========================
def is_chinese_holiday_approx(date: datetime) -> int:
    """近似判断是否为中国法定节假日。"""
    m, d = date.month, date.day
    if m == 1 and d == 1: return 1
    spring_start = 21 + (date.year % 3)
    if m == 1 and d >= spring_start: return 1
    if m == 2 and d <= spring_start - 21 + 7: return 1
    if m == 4 and 4 <= d <= 6: return 1
    if m == 5 and 1 <= d <= 3: return 1
    if m == 6 and 10 <= d <= 12: return 1
    if m == 9 and 15 <= d <= 17: return 1
    if m == 10 and 1 <= d <= 7: return 1
    return 0


# ======================== 3. 创建物资 ========================
def build_materials() -> Dict[str, Dict[str, Any]]:
    """构建电网物资主数据字典。20种物资分三种需求模式。"""
    base_materials = {
        "MAT001": {"name": "10kV油浸式变压器",   "category": "变压器",   "unit_cost": 35000.0, "lead_time": 30, "demand_pattern": "project",    "seasonal": False},
        "MAT002": {"name": "钢芯铝绞线LGJ-120",   "category": "导线",     "unit_cost": 4.5,     "lead_time": 15, "demand_pattern": "continuous",  "seasonal": True},
        "MAT003": {"name": "复合绝缘子FXBW-110",   "category": "绝缘子",   "unit_cost": 280.0,   "lead_time": 10, "demand_pattern": "sporadic",    "seasonal": False},
        "MAT004": {"name": "智能电表DTZ666",       "category": "电表",     "unit_cost": 450.0,   "lead_time": 7,  "demand_pattern": "continuous",  "seasonal": False},
        "MAT005": {"name": "10kV电缆分支箱",       "category": "配电设备", "unit_cost": 12000.0, "lead_time": 20, "demand_pattern": "project",    "seasonal": False},
    }
    extra_categories = ["变压器", "导线", "绝缘子", "电表", "开关柜", "电缆", "塔材", "金具", "保护装置", "电容器"]
    _state = np.random.get_state()
    np.random.seed(123)
    for i in range(6, NUM_MATERIALS + 1):
        code = f"MAT{i:03d}"
        cat = np.random.choice(extra_categories)
        pattern = "project" if cat in ["变压器", "开关柜"] else ("continuous" if cat in ["导线", "电缆"] else "sporadic")
        base_materials[code] = {
            "name": f"{cat}_型号{i}", "category": cat,
            "unit_cost": np.random.uniform(50, 50000),
            "lead_time": int(np.random.choice([7, 10, 15, 20, 30])),
            "demand_pattern": pattern,
            "seasonal": bool(np.random.choice([True, False], p=[0.3, 0.7])),
        }
    np.random.set_state(_state)
    return base_materials


# ======================== 4. 生成项目 ========================
def generate_projects(materials, regions, years, projects_per_year):
    """生成随机项目需求清单，每个项目有S曲线消耗规律。"""
    projects = []
    project_id = 0
    for year in years:
        for _ in range(projects_per_year):
            start = datetime(year, np.random.randint(1, 13), np.random.randint(1, 28))
            duration = np.random.randint(5, 45)
            end = start + timedelta(days=duration)
            projects.append({
                "id": f"P{year}_{project_id}", "start": start, "end": end,
                "region": np.random.choice(regions),
                "material_code": np.random.choice(list(materials.keys())),
                "total_qty": np.random.randint(80, 1200),
            })
            project_id += 1
    return projects


# ======================== 5. 日度需求仿真引擎 ========================
def simulate_daily_demand(dates, regions, materials, projects, daily_feat_map=None):
    """
    核心日度需求仿真引擎。
    🔧 优化点：强化宏观/招标/温度等因果调制系数，使特征-需求相关性达到 0.3~0.6 合理区间。
    """
    yesterday_demand = {(r, m): 0 for r in regions for m in materials}
    active_shocks = {r: 0 for r in regions}
    records = []

    for date in dates:
        temp, is_extreme_temp = simulate_daily_weather(date)
        holiday = is_chinese_holiday_approx(date)
        is_weekend = 1 if date.weekday() >= 5 else 0
        year, month, quarter = date.year, date.month, (date.month - 1) // 3 + 1

        # ---- 从月度特征中获取当天的因果调制因子 ----
        feat_factors = daily_feat_map.get(date, {}) if daily_feat_map else {}

        # 提取关键因果因子（使用增强后的敏感度系数）
        macro_factor = 1.0
        gdp_factor = 1.0
        aging_factor = 1.0
        bidding_factor = 1.0
        copper_factor = 1.0
        load_factor = 1.0
        extreme_weather_factor = 1.0
        spring_festival_factor = 1.0

        if feat_factors:
            invest_g = feat_factors.get('macro_invest_growth', 0.06)
            macro_factor = 1.0 + (invest_g - 0.06) * CAUSAL_SENSITIVITY['macro_invest_growth']

            gdp_g = feat_factors.get('gdp_growth', 0.07)
            gdp_factor = 1.0 + (gdp_g - 0.07) * CAUSAL_SENSITIVITY['gdp_growth']

            aging_r = feat_factors.get('equipment_aging_rate', 0.15)
            aging_factor = 1.0 + (aging_r - 0.15) * CAUSAL_SENSITIVITY['equipment_aging_rate']

            bid_int = feat_factors.get('bidding_intensity', 0.2)
            bidding_factor = 1.0 + (bid_int - 0.2) * CAUSAL_SENSITIVITY['bidding_intensity']

            copper_p = feat_factors.get('copper_price_index', 55000)
            copper_factor = 1.0 + (copper_p - 55000) * CAUSAL_SENSITIVITY['copper_price_index']

            load_mw = feat_factors.get('power_load_MW', 5000)
            load_factor = 1.0 + (load_mw - 5000) * CAUSAL_SENSITIVITY['power_load_MW']

            ext_days = feat_factors.get('extreme_weather_days', 1)
            extreme_weather_factor = 1.0 + (ext_days - 1) * CAUSAL_SENSITIVITY['extreme_weather_days']

            is_sf = feat_factors.get('is_spring_festival', 0)
            if is_sf:
                spring_festival_factor = 0.6

        # 全局需求乘数（GDP×春节）
        global_multiplier = gdp_factor * spring_festival_factor

        # 更新各区域突发事件状态
        for r in regions:
            if active_shocks[r] > 0:
                active_shocks[r] -= 1
            elif np.random.rand() < SHOCK_PROBABILITY:
                active_shocks[r] = SHOCK_DURATION

        current_day_demand = {(r, m): 0 for r in regions for m in materials}

        for region in regions:
            region_mult = REGION_DEMAND_FACTOR.get(region, 1.0)

            for mat_code, mat_info in materials.items():
                demand = 0

                # -------- 1) 项目脉冲需求 --------
                for p in projects:
                    if (p['region'] == region and p['material_code'] == mat_code
                            and p['start'] <= date <= p['end']):
                        proj_duration = (p["end"] - p["start"]).days + 1
                        daily_avg = p["total_qty"] / proj_duration
                        day_in_project = (date - p['start']).days
                        weight = np.exp(-0.5 * ((day_in_project - proj_duration / 2) / (proj_duration / 4)) ** 2)
                        proj_demand = np.random.poisson(daily_avg * 1.8 * weight)
                        if is_weekend:
                            proj_demand = int(proj_demand * PROJECT_WEEKEND_FACTOR)
                        if holiday:
                            proj_demand = int(proj_demand * PROJECT_HOLIDAY_FACTOR)
                        proj_demand = int(proj_demand * macro_factor * bidding_factor)
                        demand += proj_demand

                # -------- 2) 连续运维需求 --------
                if mat_info["demand_pattern"] == "continuous":
                    if mat_info["seasonal"]:
                        if month in [6, 7, 8]: base_lambda = 8
                        elif month in [12, 1, 2]: base_lambda = 6
                        else: base_lambda = 3
                    else:
                        base_lambda = 3
                    if is_weekend or holiday:
                        base_lambda = max(1, int(base_lambda * 0.5))
                    base = np.random.poisson(base_lambda)

                    # 🔧 温度修正系数增强
                    temp_deviation = abs(temp - COMFORT_TEMP)
                    temp_mult = 1.0 + (temp_deviation // 5) * CAUSAL_SENSITIVITY['temperature']
                    base = int(base * temp_mult)

                    if is_extreme_temp:
                        base = int(base * (EXTREME_HOT_FACTOR if temp > 35 else EXTREME_COLD_FACTOR))
                    base = int(base * aging_factor * load_factor * extreme_weather_factor)
                    demand += base

                # -------- 3) 零星需求 --------
                elif mat_info["demand_pattern"] == "sporadic":
                    if np.random.rand() < 0.20:
                        spo = np.random.poisson(5)
                        if is_extreme_temp: spo = int(spo * 2.0)
                        if is_weekend: spo = int(spo * 0.5)
                        if holiday: spo = int(spo * 0.5)
                        spo = int(spo * aging_factor * extreme_weather_factor)
                        demand += spo

                # -------- 后处理 --------
                demand = int(demand * region_mult * global_multiplier)
                if active_shocks[region] > 0 and mat_info["category"] in SHOCK_CATEGORIES:
                    demand = int(demand * SHOCK_MULTIPLIER)
                if demand == 0 and (not is_weekend) and (not holiday) and np.random.rand() < 0.05:
                    demand = np.random.poisson(1)

                current_day_demand[(region, mat_code)] = demand

            # -------- BOM联动（仅工作日）--------
            related_add = {}
            for mat_code in materials:
                core_demand = current_day_demand[(region, mat_code)]
                if core_demand <= 0: continue
                if (not is_weekend) and (not holiday) and (mat_code in MATERIAL_CORRELATION):
                    for related_mat in MATERIAL_CORRELATION[mat_code]:
                        if np.random.rand() < CORRELATION_PROB:
                            add_qty = int(core_demand * np.random.uniform(CORRELATION_RATIO_MIN, CORRELATION_RATIO_MAX))
                            if add_qty > 0:
                                related_add[(region, related_mat)] = related_add.get((region, related_mat), 0) + add_qty
            for (reg, mat), add_qty in related_add.items():
                current_day_demand[(reg, mat)] += add_qty

        # -------- 差异化AR平滑 --------
        for (region, mat_code), base_demand in current_day_demand.items():
            mat_info = materials[mat_code]
            if mat_info["demand_pattern"] == "continuous": ar_coeff = AR_COEFF_CONTINUOUS
            elif mat_info["demand_pattern"] == "project": ar_coeff = AR_COEFF_PROJECT
            else: ar_coeff = AR_COEFF_SPORADIC

            yesterday = yesterday_demand.get((region, mat_code), 0)
            final_demand = int(ar_coeff * yesterday + (1 - ar_coeff) * base_demand)

            if final_demand > 0:
                records.append({
                    "date": date, "year": year, "month": month, "quarter": quarter,
                    "region": region, "material_code": mat_code,
                    "category": mat_info["category"],
                    "demand_pattern": mat_info["demand_pattern"],
                    "demand_qty": final_demand,
                    "temperature": temp, "is_extreme_temp": is_extreme_temp,
                    "is_holiday": holiday, "is_weekend": is_weekend,
                })
                yesterday_demand[(region, mat_code)] = final_demand
            else:
                yesterday_demand[(region, mat_code)] = 0

    return pd.DataFrame(records)


# ======================== 6. 日度→月度聚合 ========================
def aggregate_to_monthly(df_daily, regions, materials):
    """将日度需求聚合为月度需求。输出两层：汇总层面 + 单物资层面。"""
    df_daily['year_month'] = df_daily['date'].dt.to_period('M')
    monthly_total = df_daily.groupby('year_month')['demand_qty'].sum().reset_index()
    monthly_total.columns = ['year_month', 'demand']
    monthly_total['date'] = monthly_total['year_month'].dt.to_timestamp('M')
    monthly_total = monthly_total.drop(columns='year_month')

    monthly_by_sku = df_daily.groupby(['year_month', 'region', 'material_code', 'category', 'demand_pattern']) \
                             .agg(demand=('demand_qty', 'sum'),
                                  avg_temperature=('temperature', 'mean'),
                                  extreme_days=('is_extreme_temp', 'sum'),
                                  holiday_days=('is_holiday', 'sum'),
                                  weekend_days=('is_weekend', 'sum')).reset_index()
    monthly_by_sku['date'] = monthly_by_sku['year_month'].dt.to_timestamp('M')
    monthly_by_sku = monthly_by_sku.drop(columns='year_month')

    return monthly_total, monthly_by_sku


# ======================== 7. 生成月度外部特征 ========================
def generate_monthly_features(monthly_dates):
    """生成月度外部特征。保持独立随机生成，与需求分离。"""
    T = len(monthly_dates)
    months = np.array([d.month for d in monthly_dates])
    t = np.arange(T)

    temperature = 12 + 14 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 2.5, T)
    extreme_weather_days = np.random.poisson(lam=1.2, size=T)
    wind_speed = 4.5 + 2.0 * np.sin(2 * np.pi * months / 12 + 1.5) + np.random.normal(0, 0.8, T)

    macro_invest_growth = 0.06 + 0.015 * np.sin(2 * np.pi * t / 36) + np.random.normal(0, 0.008, T)
    gdp_growth = 0.07 + 0.01 * np.sin(2 * np.pi * t / 48) + np.random.normal(0, 0.005, T)
    copper_price = 55000 + 3000 * np.sin(2 * np.pi * t / 24) + np.cumsum(np.random.normal(0, 500, T))

    power_load = 5000 + 800 * gdp_growth * t + 600 * np.sin(2 * np.pi * months / 12) \
                 + 200 * (temperature - 12) ** 2 / 196 + np.random.normal(0, 150, T)
    equipment_aging_rate = 0.15 + 0.005 * t + 0.05 * (t % 36 < 3).astype(float) + np.random.normal(0, 0.01, T)
    equipment_aging_rate = np.clip(equipment_aging_rate, 0, 1)

    bidding_intensity = np.where(np.isin(months, [3, 6, 9, 12]),
                                 1.0 + np.random.uniform(0, 0.4, T),
                                 0.2 + np.random.uniform(0, 0.1, T))
    renewable_capacity = 100 + 800 / (1 + np.exp(-0.2 * (t - T / 2))) + np.random.normal(0, 30, T)
    is_spring_festival = np.where(np.isin(months, [1, 2]), 1, 0)

    features = pd.DataFrame({
        'date': monthly_dates,
        'temperature': temperature,
        'extreme_weather_days': extreme_weather_days,
        'wind_speed': wind_speed,
        'macro_invest_growth': macro_invest_growth,
        'gdp_growth': gdp_growth,
        'copper_price_index': copper_price,
        'power_load_MW': power_load,
        'equipment_aging_rate': equipment_aging_rate,
        'bidding_intensity': bidding_intensity,
        'renewable_capacity_MW': renewable_capacity,
        'is_spring_festival': is_spring_festival,
    })
    return features


# ======================== 8. 月度特征→日度映射辅助函数 ========================
def _map_monthly_to_daily(monthly_features_df, daily_dates):
    """将月度外部特征映射到日度。"""
    monthly_features_df = monthly_features_df.copy()
    monthly_features_df['year_month'] = monthly_features_df['date'].dt.to_period('M')
    month_map = {}
    for _, row in monthly_features_df.iterrows():
        key = row['year_month']
        month_map[key] = row.drop(['date', 'year_month']).to_dict()

    daily_feat_map = {}
    for d in daily_dates:
        ym = pd.Period(d, freq='M')
        if ym in month_map:
            daily_feat_map[d] = month_map[ym]
        else:
            closest = min(month_map.keys(), key=lambda x: abs(x - ym))
            daily_feat_map[d] = month_map[closest]
    return daily_feat_map


# ======================== 9. 主生成函数 ========================
def generate_grid_material_demand(
        start='1995-01-01', end='2025-12-31', freq='ME',
        seed=42, add_missing=True, add_outliers=True,
        save_individual=True,
):
    """
    电网物资需求仿真数据主生成函数（日度仿真→月度聚合）。
    🔧 优化点：
      1. 移除冗余的第一次无特征仿真，统一执行因果链路
      2. 注入 5% 月度零需求，贴合业务实际
      3. 末尾追加轻量级质量诊断，便于即时验证
    """
    np.random.seed(seed)
    random.seed(seed)

    SEP = "=" * 60
    print(SEP)
    print("  电网物资需求仿真数据生成引擎 v2.1")
    print("  策略: 日度仿真 → 月度聚合")
    print(SEP)

    # ============ 1. 物资与项目 ============
    materials = build_materials()
    daily_dates = pd.date_range(start=start, end=end, freq='D')
    years = sorted(set(d.year for d in daily_dates))
    projects = generate_projects(materials, REGIONS, years, PROJECTS_PER_YEAR)

    n_project_type = sum(1 for m in materials.values() if m['demand_pattern'] == 'project')
    n_continuous_type = sum(1 for m in materials.values() if m['demand_pattern'] == 'continuous')
    n_sporadic_type = sum(1 for m in materials.values() if m['demand_pattern'] == 'sporadic')

    print(f"\n  物资: {len(materials)}种 (项目型{n_project_type} / 连续型{n_continuous_type} / 零星型{n_sporadic_type})")
    print(f"  区域: {len(REGIONS)}个 {REGIONS}")
    print(f"  仿真: {len(daily_dates)}天 ({start} → {end})")
    print(f"  项目: {len(projects)}个 (每年{PROJECTS_PER_YEAR}个)")

    # ============ 2. 生成月度外部特征 & 映射到日度 ============
    print(f"\n  [1/4] 生成外部特征并映射至日度...")
    monthly_dates = pd.date_range(start=start, end=end, freq=freq)
    features_df = generate_monthly_features(monthly_dates)
    daily_feat_map = _map_monthly_to_daily(features_df, daily_dates)

    # ============ 3. 日度需求仿真（注入因果调制） ============
    print(f"  [2/4] 日度需求仿真中（增强因果链）...")
    df_daily = simulate_daily_demand(daily_dates, REGIONS, materials, projects, daily_feat_map)

    total_possible = len(daily_dates) * len(REGIONS) * len(materials)
    zero_ratio_daily = 1 - len(df_daily) / total_possible
    print(f"        日度有需求记录: {len(df_daily):,}条")
    print(f"        日度零需求比例: {zero_ratio_daily:.1%}")

    # ============ 4. 月度聚合 ============
    print(f"  [3/4] 聚合为月度数据...")
    monthly_total, monthly_by_sku = aggregate_to_monthly(df_daily, REGIONS, materials)

    T = len(monthly_dates)
    demand_by_month = dict(zip(monthly_total['date'], monthly_total['demand']))
    demand_full = np.array([demand_by_month.get(d, 0) for d in monthly_dates], dtype=float)

    # 🔧 核心修复：注入 ~5% 零需求月（模拟项目空窗期/物资停用）
    zero_mask = np.random.rand(T) < 0.05
    demand_full[zero_mask] = 0.0

    zero_ratio_monthly = np.mean(demand_full == 0)
    print(f"        月度数据点: {T}个月")
    print(f"        🔧 月度零需求比例: {zero_ratio_monthly:.1%} (目标: ~5%)")
    print(f"        月度需求均值: {demand_full.mean():.1f}")

    # ============ 5. 合并 & 注入噪声 ============
    df = features_df.copy()
    df['demand'] = demand_full

    print(f"  [4/4] 注入缺失值/异常值...")
    if add_missing:
        non_zero_mask = df['demand'] > 0
        missing_candidates = np.where(non_zero_mask)[0]
        n_missing = max(1, int(0.03 * T))
        if len(missing_candidates) >= n_missing:
            miss_idx = np.random.choice(missing_candidates, size=n_missing, replace=False)
            cont_cols = ['demand', 'temperature', 'wind_speed', 'copper_price_index']
            df.iloc[miss_idx, df.columns.isin(cont_cols)] = np.nan
    if add_outliers:
        non_zero_mask = df['demand'] > 0
        outlier_candidates = np.where(non_zero_mask)[0]
        n_outliers = max(1, int(0.015 * T))
        if len(outlier_candidates) >= n_outliers:
            out_idx = np.random.choice(outlier_candidates, size=n_outliers, replace=False)
            df.loc[df.index[out_idx], 'demand'] *= np.random.choice([0.3, 3.5], size=n_outliers)

    # ============ 6. 保存单物资月度数据 ============
    if save_individual:
        os.makedirs(MATERIAL_DEMAND_PATH, exist_ok=True)
        for mat_code in materials:
            mat_monthly = monthly_by_sku[monthly_by_sku['material_code'] == mat_code].copy()
            for region in REGIONS:
                sku_data = mat_monthly[mat_monthly['region'] == region].copy()
                if len(sku_data) > 0:
                    filename = f"{mat_code}_{region}_monthly.csv"
                    sku_data.to_csv(os.path.join(MATERIAL_DEMAND_PATH, filename), index=False)
        print(f"        单物资数据: {len(materials) * len(REGIONS)}个SKU已保存至 {MATERIAL_DEMAND_PATH}")

    # ============ 7. 质量诊断（快速验证） ============
    print(f"\n{SEP}")
    print(f"  📊 即时质量诊断")
    print(f"  总月数:        {T}")
    print(f"  零需求月数:    {int(np.sum(df['demand'] == 0))} ({np.mean(df['demand'] == 0):.1%})")
    print(f"  需求均值:      {df['demand'].mean():.1f} | 标准差: {df['demand'].std():.1f}")
    print(f"  需求范围:      [{df['demand'].min():.0f}, {df['demand'].max():.0f}]")
    acf_1 = df['demand'].dropna().autocorr(lag=1)
    acf_12 = df['demand'].dropna().autocorr(lag=12)
    print(f"  自相关性:      lag-1={acf_1:.3f}, lag-12={acf_12:.3f}")
    print(f"{SEP}")

    return df.set_index('date')


# ======================== 主入口 ========================
if __name__ == '__main__':
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df_sim = generate_grid_material_demand()
    df_sim.to_csv(SIM_DATASET_PATH)
    print(f"\n  ✅ 主文件已保存: {SIM_DATASET_PATH}")
    print(f"  📐 数据形状: {df_sim.shape}")