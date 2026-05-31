# 在 Python 中执行，或手动在文件管理器中删除
import os, glob

# 1. 删除旧数据 CSV
csv_path = "data/raw/grid_material_003.csv"
if os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"已删除: {csv_path}")

# 2. 删除 VMD 缓存
for f in glob.glob("figures/data/003/vmd_cache_*.npz"):
    os.remove(f)
    print(f"已删除: {f}")

# 3. 删除旧模型权重
for f in glob.glob("models/003/*.pth"):
    os.remove(f)
    print(f"已删除: {f}")

# 4. 删除旧特征筛选器
for f in glob.glob("figures/data/003/feature_selector_state*.json"):
    os.remove(f)
    print(f"已删除: {f}")

# 5. 删除旧实验结果
for f in glob.glob("figures/data/003/*_results*.csv"):
    os.remove(f)
    print(f"已删除: {f}")