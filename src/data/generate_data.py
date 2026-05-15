import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from configs.config import SIM_DATASET_PATH, DATA_RAW_DIR


def generate_grid_material_demand(
        start='2005-01-01', end='2025-12-31', freq='ME',
        base_demand=500, noise_std=35, seed=42, add_missing=True, add_outliers=True
):
    np.random.seed(seed)
    dates = pd.date_range(start, end, freq=freq)
    T = len(dates)
    months = np.array([d.month for d in dates])
    t = np.arange(T)

    # ==================== 1. 协变量 (特征) 生成 ====================

    # 1.1 气象与环境
    temperature = 12 + 14 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 2.5, T)
    extreme_weather_days = np.random.poisson(lam=1.2, size=T)
    wind_speed = 4.5 + 2.0 * np.sin(2 * np.pi * months / 12 + 1.5) + np.random.normal(0, 0.8, T)

    # 1.2 宏观与市场
    macro_invest_growth = 0.06 + 0.015 * np.sin(2 * np.pi * t / 36) + np.random.normal(0, 0.008, T)
    gdp_growth = 0.07 + 0.01 * np.sin(2 * np.pi * t / 48) + np.random.normal(0, 0.005, T)
    # 铜价随机游走
    copper_price = 55000 + 3000 * np.sin(2 * np.pi * t / 24) + np.cumsum(np.random.normal(0, 500, T))

    # 1.3 生产与运行状态
    # 用电负荷 (受GDP和极端温度双重驱动)
    power_load = 5000 + 800 * gdp_growth * t + 600 * np.sin(2 * np.pi * months / 12) \
                 + 200 * (temperature - 12) ** 2 / 196 + np.random.normal(0, 150, T)  # 温度偏离度越大，负荷越高
    # 设备老化率 (随时间上升，每3年有一次旧批次集中到期跃升)
    equipment_aging_rate = 0.15 + 0.005 * t + 0.05 * (t % 36 < 3).astype(float) + np.random.normal(0, 0.01, T)
    equipment_aging_rate = np.clip(equipment_aging_rate, 0, 1)

    # 1.4 项目与政策
    # 招标强度 (季度末脉冲)
    bidding_intensity = np.where(np.isin(months, [3, 6, 9, 12]),
                                 1.0 + np.random.uniform(0, 0.4, T),
                                 0.2 + np.random.uniform(0, 0.1, T))
    # 新能源并网容量 (S型政策驱动)
    renewable_capacity = 100 + 800 / (1 + np.exp(-0.2 * (t - T / 2))) + np.random.normal(0, 30, T)
    # 春节停工标记
    is_spring_festival = np.where(np.isin(months, [1, 2]), 1, 0)

    # ==================== 2. 需求目标变量 生成 ====================
    # 核心：让 demand 成为特征的复杂非线性函数

    # 2.1 基础趋势与宏观驱动 (投资越高，需求越大)
    T_t = base_demand * (1 + macro_invest_growth * 2)

    # 2.2 季节与气象驱动 (温度极端+极端天气+大风 = 抢修需求激增)
    # 温度偏离舒适区的程度
    temp_deviation = np.abs(temperature - 12)
    # 极端天气对需求的非线性放大
    S_t = 100 * np.sin(2 * np.pi * months / 12) + 80 * temp_deviation \
          + 150 * np.tanh(extreme_weather_days / 2) + 20 * np.maximum(wind_speed - 6, 0)

    # 2.3 老化与运维驱动 (老化率越高，更换需求越大；老化跃升期需求脉冲)
    A_t = 2000 * equipment_aging_rate ** 2 + 100 * (t % 36 < 3).astype(float)

    # 2.4 基建项目驱动 (招标季度需求高，但受铜价抑制)
    # 铜价越高，项目采购越谨慎（抑制系数 0.7~1.0）
    price_suppression = 1.0 - 0.3 * (copper_price - 55000) / 8000
    price_suppression = np.clip(price_suppression, 0.7, 1.0)
    C_t = 350 * bidding_intensity * price_suppression

    # 2.5 新能源配套驱动 (新能源装机增量带动并网物资需求)
    renewable_increment = np.concatenate(([renewable_capacity[0]], np.diff(renewable_capacity)))
    R_t = 2.5 * np.maximum(renewable_increment, 0)

    # 2.6 特殊事件 (春节停工)
    F_t = -150 * is_spring_festival

    # 2.7 稀疏高幅值冲击 (突发重大政策/事故)
    event_indices = np.random.choice(T, size=max(1, int(T * 0.015)), replace=False)
    E_t = np.zeros(T)
    E_t[event_indices] = np.random.choice([-0.2, 0.3, 0.5], size=len(event_indices)) * base_demand

    # 2.8 噪声
    epsilon_t = np.random.normal(0, noise_std, T)

    # ==================== 3. 合成目标变量 ====================
    demand = T_t + S_t + A_t + C_t + R_t + F_t + E_t + epsilon_t
    demand = np.maximum(demand, 0)  # 需求非负约束

    # ==================== 4. 整合数据 ====================
    df = pd.DataFrame({
        'date': dates,
        'demand': demand,
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
        'is_spring_festival': is_spring_festival
    })

    # 注入缺失值与异常值
    if add_missing:
        mask = np.random.rand(T) < 0.03
        cont_cols = ['demand', 'temperature', 'wind_speed', 'copper_price_index']
        df.loc[mask, df.columns.isin(cont_cols)] = np.nan
    if add_outliers:
        out_idx = np.random.choice(T, size=int(0.015 * T), replace=False)
        df.loc[out_idx, 'demand'] *= np.random.choice([0.3, 3.5], size=len(out_idx))

    return df.set_index('date')


if __name__ == '__main__':
    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        print(f"📂 已创建目录: {DATA_RAW_DIR}")

    df_sim = generate_grid_material_demand()
    df_sim.to_csv(SIM_DATASET_PATH)

    print(f"✅ 高保真仿真数据已成功保存至: {SIM_DATASET_PATH}")
    print(f"   数据形状: {df_sim.shape}")
    print(f"   包含特征: {list(df_sim.columns)}")