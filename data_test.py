import numpy as np
import pandas as pd
from vmdpy import VMD

# 加载数据
df = pd.read_csv('data/raw/grid_material_003.csv', parse_dates=['date'], index_col='date')

# ★ 关键修正：线性插值替代 dropna，保持均匀采样 ★
# dropna() 会删除NaN行，破坏时间序列等间距，
# 导致VMD无法识别周期结构（6月谐波变成4月伪周期）
demand = df['demand'].interpolate(method='linear').values

# 基本诊断
demand_series = pd.Series(demand)
print(f"均值: {demand.mean():.0f}")
print(f"CV: {demand.std()/demand.mean():.2%}")
print(f"lag-1: {demand_series.autocorr(1):.3f}")
print(f"lag-12: {demand_series.autocorr(12):.3f}")
print(f"lag-38: {demand_series.autocorr(38):.3f}")

# VMD 分解（参数与主模型一致）
u, u_hat, omega = VMD(demand, alpha=2000, tau=0, K=4, DC=0, init=1, tol=1e-7)

# ★ omega 是 (n_iter, K) 的2D数组，取最后一行 ★
omega_final = omega[-1] if omega.ndim > 1 else omega
print(f"\n最终中心频率: {omega_final}")
print(f"对应周期: {1/omega_final} 月")

total_var = np.var(demand)
for k in range(4):
    pct = np.var(u[k]) / total_var * 100
    print(f"IMF{k+1}: f={omega_final[k]:.6f} (T={1/omega_final[k]:.1f}月), var%={pct:.1f}%")

# ==================== 验收断言 ====================
print(f"\n{'='*50}")
print("VMD 频谱验收")
print(f"{'='*50}")

checks = [
    (omega_final[0] < 0.005, f"IMF1 近直流: f={omega_final[0]:.6f}", "f < 0.005"),
    (0.080 < omega_final[1] < 0.086, f"IMF2 年周期: f={omega_final[1]:.6f}", "0.080-0.086"),
    (0.160 < omega_final[2] < 0.170, f"IMF3 半年谐波: f={omega_final[2]:.6f}", "0.160-0.170"),
    (np.var(u[1])/total_var > 0.25, f"IMF2 方差: {np.var(u[1])/total_var:.1%}", ">25%"),
    (np.var(u[2])/total_var > 0.05, f"IMF3 方差: {np.var(u[2])/total_var:.1%}", ">5%"),
]

all_pass = True
for passed, desc, criteria in checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {desc} (要求: {criteria})")
    if not passed:
        all_pass = False

if all_pass:
    print("\n✅ VMD 频谱验收全部通过！")
else:
    print("\n❌ VMD 频谱验收未通过，请检查数据生成参数。")