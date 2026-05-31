"""
电网物资需求仿真数据生成引擎

核心改动：
1. 只模拟一种物资（变压器，项目型）和一个区域（A），生成总需求序列
2. 去除物资循环、区域循环、BOM联动等多维度聚合逻辑
3. 保持日度仿真→月度聚合的基本结构，确保业务真实性
4. 后处理增强保证需求范围合理（均值~4000）且特征相关性显著
5. 输出格式：仅包含 date, demand 和 11个外部特征，无任何分类列

输出文件：
    grid_material_demand_sim.csv : 月度需求 + 外部特征
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from configs.config import (
    SIM_DATASET_PATH,
    DATA_RAW_DIR,
)

# ==================== 全局参数配置 ====================
RANDOM_SEED = 42

# 固定物资和区域
MATERIAL_CODE = "MAT001"
MATERIAL_NAME = "10kV油浸式变压器"
CATEGORY = "变压器"
DEMAND_PATTERN = "project"      # 项目脉冲型
SEASONAL = False

REGION = "A"
REGION_DEMAND_FACTOR = 1.0      # 区域乘数为1

# 项目参数
PROJECTS_PER_YEAR_BASE = 15
PROJECTS_GROWTH_RATE = 0.3

# 因果敏感度系数（与v3.1相同）
CAUSAL_SENSITIVITY = {
    'macro_invest_growth': 80.0,
    'bidding_intensity': 8.0,
    'equipment_aging_rate': 12.0,
    'temperature': 0.6,
    'gdp_growth': 35.0,
    'copper_price_index': -0.00003,
    'power_load_MW': 0.0001,
    'extreme_weather_days': 0.5,
}

# 自回归系数（项目型）
AR_COEFF_PROJECT = 0.6

# 外部因素系数
PROJECT_WEEKEND_FACTOR = 0.75
PROJECT_HOLIDAY_FACTOR = 0.60
EXTREME_HOT_FACTOR = 2.0
EXTREME_COLD_FACTOR = 1.5
COMFORT_TEMP = 20.0

# 突发事件冲击
SHOCK_PROBABILITY = 0.001
SHOCK_MULTIPLIER = 2.0
SHOCK_DURATION = 5
SHOCK_CATEGORIES = {"变压器"}   # 仅匹配当前物资类别

# 异常值参数
OUTLIER_RATIO = 0.005
OUTLIER_FACTORS = [0.7, 1.5]

# 月度后处理参数
LONG_TREND_SLOPE = 6          # 每年增长约72单位
SEASONAL_AMPLITUDE = 200
TARGET_MEAN = 4000


# ======================== 天气与节假日 ========================
def simulate_daily_weather(date: datetime) -> tuple:
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

def is_chinese_holiday_approx(date: datetime) -> int:
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


# ======================== 项目生成 ========================
def generate_projects_with_trend(years, base_projects_per_year, growth_rate):
    """生成项目清单，仅针对固定物资和区域"""
    projects = []
    project_id = 0
    for idx, year in enumerate(years):
        n_proj = int(base_projects_per_year + idx * growth_rate)
        for _ in range(n_proj):
            # 项目集中在春季和秋季
            month_probs = [1,1,2,2,2,1,1,1,2,2,2,1]
            month = np.random.choice(range(1,13), p=np.array(month_probs)/sum(month_probs))
            start = datetime(year, month, np.random.randint(1, 28))
            duration = np.random.randint(20, 60)
            end = start + timedelta(days=duration)
            total_qty = np.random.randint(80, 500)
            projects.append({
                "id": f"P{year}_{project_id}",
                "start": start, "end": end,
                "total_qty": total_qty,
            })
            project_id += 1
    return projects


# ======================== 日度需求仿真 ========================
def simulate_daily_demand(dates, projects, daily_feat_map):
    """仿真单一物资在固定区域的日度需求"""
    yesterday_demand = 0
    active_shocks = 0
    records = []

    for date in dates:
        temp, is_extreme_temp = simulate_daily_weather(date)
        holiday = is_chinese_holiday_approx(date)
        is_weekend = 1 if date.weekday() >= 5 else 0
        year, month, quarter = date.year, date.month, (date.month - 1) // 3 + 1

        feat_factors = daily_feat_map.get(date, {}) if daily_feat_map else {}

        macro_factor = 1.0
        gdp_factor = 1.0
        aging_factor = 1.0
        bidding_factor = 1.0
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
            ext_days = feat_factors.get('extreme_weather_days', 1)
            extreme_weather_factor = 1.0 + (ext_days - 1) * CAUSAL_SENSITIVITY['extreme_weather_days']
            is_sf = feat_factors.get('is_spring_festival', 0)
            if is_sf:
                spring_festival_factor = 0.7

        global_multiplier = gdp_factor * spring_festival_factor

        # 更新突发事件
        if active_shocks > 0:
            active_shocks -= 1
        elif np.random.rand() < SHOCK_PROBABILITY:
            active_shocks = SHOCK_DURATION

        demand = 0

        # 项目脉冲
        for p in projects:
            if p['start'] <= date <= p['end']:
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

        # 后处理
        demand = int(demand * REGION_DEMAND_FACTOR * global_multiplier)
        if active_shocks > 0 and CATEGORY in SHOCK_CATEGORIES:
            demand = int(demand * SHOCK_MULTIPLIER)
        if demand == 0 and not is_weekend and not holiday and np.random.rand() < 0.02:
            demand = np.random.poisson(2)

        # AR平滑
        final_demand = int(AR_COEFF_PROJECT * yesterday_demand + (1 - AR_COEFF_PROJECT) * demand)

        if final_demand > 0:
            records.append({
                "date": date, "year": year, "month": month, "quarter": quarter,
                "demand_qty": final_demand,
                "temperature": temp, "is_extreme_temp": is_extreme_temp,
                "is_holiday": holiday, "is_weekend": is_weekend,
            })
            yesterday_demand = final_demand
        else:
            yesterday_demand = 0

    return pd.DataFrame(records)


# ======================== 聚合与外部特征 ========================
def aggregate_to_monthly(df_daily):
    df_daily['year_month'] = df_daily['date'].dt.to_period('M')
    monthly_total = df_daily.groupby('year_month')['demand_qty'].sum().reset_index()
    monthly_total.columns = ['year_month', 'demand_raw']
    monthly_total['date'] = monthly_total['year_month'].dt.to_timestamp('M')
    monthly_total = monthly_total.drop(columns='year_month')
    return monthly_total

def generate_monthly_features(monthly_dates):
    T = len(monthly_dates)
    months = np.array([d.month for d in monthly_dates])
    t = np.arange(T)
    temperature = 12 + 14 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 2, T)
    extreme_weather_days = np.random.poisson(lam=1.0, size=T)
    wind_speed = 4.5 + 2.0 * np.sin(2 * np.pi * months / 12 + 1.5) + np.random.normal(0, 0.8, T)
    macro_invest_growth = 0.06 + 0.015 * np.sin(2 * np.pi * t / 36) + 0.003 * (t / T) + np.random.normal(0, 0.002, T)  # 原0.005
    gdp_growth = 0.07 + 0.01 * np.sin(2 * np.pi * t / 48) + 0.002 * (t / T) + np.random.normal(0, 0.0015, T)  # 原0.003
    copper_price = 55000 + 3000 * np.sin(2 * np.pi * t / 24) + np.cumsum(np.random.normal(0, 300, T))
    power_load = 5000 + 800 * gdp_growth * t + 600 * np.sin(2 * np.pi * months / 12) \
                 + 200 * (temperature - 12) ** 2 / 196 + np.random.normal(0, 100, T)
    equipment_aging_rate = 0.15 + 0.005 * t + 0.05 * (t % 36 < 3).astype(float) + np.random.normal(0, 0.008, T)
    equipment_aging_rate = np.clip(equipment_aging_rate, 0, 1)
    bidding_intensity = np.where(np.isin(months, [3,6,9,12]),
                                 0.8 + np.random.uniform(0, 0.3, T),
                                 0.2 + np.random.uniform(0, 0.1, T))
    renewable_capacity = 100 + 800 / (1 + np.exp(-0.2 * (t - T/2))) + np.random.normal(0, 30, T)
    is_spring_festival = np.where(np.isin(months, [1,2]), 1, 0)
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

def _map_monthly_to_daily(monthly_features_df, daily_dates):
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


# ======================== 主生成函数 ========================
def generate_grid_material_demand(
        start='1995-01-01', end='2025-12-31', freq='ME',
        seed=42, add_missing=True, add_outliers=True,
):
    np.random.seed(seed)
    random.seed(seed)

    SEP = "=" * 60
    print(SEP)
    print("  电网物资需求仿真数据生成引擎 v4.0 (单一物资/区域版本)")
    print("  策略: 日度仿真 → 月度聚合 + 后处理增强")
    print(SEP)

    # 项目生成
    daily_dates = pd.date_range(start=start, end=end, freq='D')
    years = sorted(set(d.year for d in daily_dates))
    projects = generate_projects_with_trend(years, PROJECTS_PER_YEAR_BASE, PROJECTS_GROWTH_RATE)

    print(f"\n  物资: {MATERIAL_NAME}")
    print(f"  区域: {REGION}")
    print(f"  仿真: {len(daily_dates)}天 ({start} → {end})")
    print(f"  项目: {len(projects)}个 (年均约{len(projects)//len(years):.0f}个)")

    # 外部特征
    print(f"\n  [1/4] 生成外部特征并映射至日度...")
    monthly_dates = pd.date_range(start=start, end=end, freq=freq)
    features_df = generate_monthly_features(monthly_dates)
    daily_feat_map = _map_monthly_to_daily(features_df, daily_dates)

    # 日度仿真
    print(f"  [2/4] 日度需求仿真中...")
    df_daily = simulate_daily_demand(daily_dates, projects, daily_feat_map)

    # 聚合
    print(f"  [3/4] 聚合为月度数据并增强可预测性...")
    monthly_total = aggregate_to_monthly(df_daily)
    T = len(monthly_dates)
    demand_by_month = dict(zip(monthly_total['date'], monthly_total['demand_raw']))
    demand_raw = np.array([demand_by_month.get(d, 0) for d in monthly_dates], dtype=float)

    # ========== 后处理增强（优化季节与相关性）==========
    t = np.arange(T)
    months = np.array([d.month for d in monthly_dates])

    # 1. 降低趋势斜率，避免破坏季节模式
    trend = 3 * t / 12

    # 2. 增大季节幅度，并调整相位使夏季需求最高（原相位 -π/2 可能偏移，改用余弦）
    seasonal = 350 * np.cos(2 * np.pi * (months - 7) / 12)  # 7月峰值

    # 3. 外部特征贡献权重优化
    macro = features_df['macro_invest_growth'].values
    gdp = features_df['gdp_growth'].values
    temp = features_df['temperature'].values
    bidding = features_df['bidding_intensity'].values

    external_contribution = (
            (macro - 0.06) * 1500  # 增强宏观投资影响
            + (gdp - 0.07) * 1200  # 显著增强 GDP 贡献
            + (temp - 12) * 12
            + (bidding - 0.2) * 800  # 增强招标强度影响
    )
    demand_base = trend + seasonal + external_contribution + 2000

    # 4. 提高外部特征融合权重（更多依赖强因果部分）
    demand = 0.2 * demand_raw + 0.8 * demand_base
    demand = np.clip(demand, 200, 10000)

    # 5. 降低AR平滑系数，保留季节性波动（原0.6）
    alpha = 0.35
    for i in range(1, len(demand)):
        demand[i] = alpha * demand[i - 1] + (1 - alpha) * demand[i]

    # 6. 全局缩放至目标均值
    scale = 4000 / demand.mean()
    demand = demand * scale
    demand = np.clip(demand, 100, 12000)

    # 7. 极少量零值（<1%）
    zero_mask = np.random.rand(T) < 0.01
    demand[zero_mask] = 0

    df = features_df.copy()
    df['demand'] = demand
    print(f"        月度数据点: {T}个月")
    print(f"        月度零需求比例: {np.mean(demand == 0):.1%}")
    print(f"        月度需求均值: {demand.mean():.1f}")

    # 注入缺失值与异常值
    print(f"  [4/4] 注入缺失值/异常值（温和）...")
    if add_missing:
        non_zero_mask = df['demand'] > 0
        missing_candidates = np.where(non_zero_mask)[0]
        n_missing = max(1, int(0.02 * T))
        if len(missing_candidates) >= n_missing:
            miss_idx = np.random.choice(missing_candidates, size=n_missing, replace=False)
            cont_cols = ['demand', 'temperature', 'wind_speed', 'copper_price_index']
            df.iloc[miss_idx, df.columns.isin(cont_cols)] = np.nan
    if add_outliers:
        non_zero_mask = df['demand'] > 0
        outlier_candidates = np.where(non_zero_mask)[0]
        n_outliers = max(1, int(OUTLIER_RATIO * T))
        if len(outlier_candidates) >= n_outliers:
            out_idx = np.random.choice(outlier_candidates, size=n_outliers, replace=False)
            df.loc[df.index[out_idx], 'demand'] *= np.random.choice(OUTLIER_FACTORS, size=n_outliers)

    # 质量诊断
    print(f"\n{SEP}")
    print(f"  📊 即时质量诊断")
    print(f"  总月数:        {T}")
    print(f"  零需求月数:    {int(np.sum(df['demand'] == 0))} ({np.mean(df['demand'] == 0):.1%})")
    print(f"  需求均值:      {df['demand'].mean():.1f} | 标准差: {df['demand'].std():.1f}")
    print(f"  需求范围:      [{df['demand'].min():.0f}, {df['demand'].max():.0f}]")
    acf_1 = df['demand'].dropna().autocorr(lag=1)
    acf_12 = df['demand'].dropna().autocorr(lag=12)
    print(f"  自相关性:      lag-1={acf_1:.3f}, lag-12={acf_12:.3f}")
    corr_list = []
    for col in ['macro_invest_growth', 'gdp_growth', 'temperature', 'bidding_intensity']:
        if col in df.columns:
            corr = df[col].corr(df['demand'])
            corr_list.append(f"{col}:{corr:.2f}")
    print(f"  特征-需求相关性: {', '.join(corr_list)}")
    print(f"{SEP}")

    return df.set_index('date')


if __name__ == '__main__':
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    df_sim = generate_grid_material_demand()
    df_sim.to_csv(SIM_DATASET_PATH)
    print(f"\n  ✅ 主文件已保存: {SIM_DATASET_PATH}")
    print(f"  📐 数据形状: {df_sim.shape}")