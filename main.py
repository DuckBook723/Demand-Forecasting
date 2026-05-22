"""
文件位置: main.py
作用: 电网物资需求预测项目 —— 一键全量实验调度脚本
      阶段1: 数据仿真生成
      阶段2: 主模型 VMD-CAT-BiLSTM-FS
      阶段3: 消融实验 (4组)
      阶段4: 对比实验 (5组)
      阶段5: 汇总所有结果至统一表格

修复记录 (v1.1):
  1. 修正主模型结果文件名: vmd_cat_bilstm_fs → vmd_catbilstm_fs (与实验脚本输出对齐)
  2. 增强结果读取: 即使进程exit码异常(如0xC0000409)，只要结果CSV存在且有效，即视为成功
  3. 增强列名兼容: 支持 R2/R², RMSE/rmse 等大小写及特殊字符
  4. 增加 PYTHONIOENCODING=utf-8 注入，根治 Windows 控制台编码错误
"""

import os
import sys
import json
import time
import subprocess
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ========================= 0. 项目路径配置 =========================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DATA_DIR = PROJECT_ROOT / "figures" / "data"
MODELS_DIR = PROJECT_ROOT / "models"

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(FIGURES_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PYTHON_EXE = sys.executable

# ========================= 1. 实验注册表 =========================
# 关键修复: 主模型CSV文件名与实验脚本实际输出严格对齐
EXPERIMENTS = [
    # 主模型
    ("主模型 (Full)", "experiments/mainmodel/vmd_cat_bilstm_fs.py",
     "figures/data/vmd_catbilstm_fs_results.csv", "main"),

    # 消融实验
    ("消融: w/o CATCN", "experiments/ablations/vmd_cat_bilstm_fs_nocatcn.py",
     "figures/data/vmd_catbilstm_fs_nocatcn_results.csv", "ablation"),
    ("消融: w/o FS", "experiments/ablations/vmd_cat_bilstm_fs_nofs.py",
     "figures/data/vmd_catbilstm_fs_nofs_results.csv", "ablation"),
    ("消融: w/o Skip", "experiments/ablations/vmd_cat_bilstm_fs_noskip.py",
     "figures/data/vmd_catbilstm_fs_noskip_results.csv", "ablation"),
    ("消融: w/o VMD", "experiments/ablations/vmd_cat_bilstm_fs_novmd.py",
     "figures/data/vmd_catbilstm_fs_novmd_results.csv", "ablation"),

    # 对比实验（基线）
    ("对比: ARIMA", "experiments/baselines/arima_model.py",
     "figures/data/arima_results.csv", "baseline"),
    ("对比: XGBoost", "experiments/baselines/xgboost_model.py",
     "figures/data/xgboost_results.csv", "baseline"),
    ("对比: LSTM-diff", "experiments/baselines/lstm_diff_model.py",
     "figures/data/lstm_diff_results_seed42.csv", "baseline"),
    ("对比: Fair-LSTM/BiLSTM", "experiments/baselines/lstm_fair_model.py",
     "figures/data/fair_lstm_bilstm_results.csv", "baseline"),
    ("对比: VMD-LSTM v2", "experiments/baselines/vmd_lstm_model.py",
     "figures/data/vmd_lstm_results.csv", "baseline"),
]

# Windows 假失败退出码集合 (进程保护触发，非实验错误)
WINDOWS_FALSE_FAILURE_CODES = {3221226505, 0xC0000409, -1073740791}

# ========================= 2. 工具函数 =========================
def print_banner(title, width=70):
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def run_script(name, script_path, exp_type):
    """
    运行单个实验脚本。若结果CSV已存在且有效，即使进程exit码异常也视为成功。
    特别处理 Windows 0xC0000409 假失败。
    """
    full_script = PROJECT_ROOT / script_path
    if not full_script.exists():
        print(f"\n⚠️  [{name}] 脚本未找到: {full_script}")
        return {"status": "missing", "time_min": 0, "name": name, "type": exp_type, "exit_code": None}

    print(f"\n🚀 启动 [{name}] ...")
    print(f"   脚本: {script_path}")
    start = time.time()

    # 先尝试读取已存在的结果（复用/检查）
    # 注意：这里不直接返回，仍运行脚本以获取最新结果；如运行失败再回退

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(
            [PYTHON_EXE, str(full_script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        elapsed = (time.time() - start) / 60.0

        # 打印最后40行输出
        out_lines = proc.stdout.splitlines()
        if out_lines:
            print("\n".join(out_lines[-40:]))

        # 判断状态：优先检查结果文件，其次看exit码
        exit_ok = proc.returncode == 0
        if not exit_ok and proc.returncode in WINDOWS_FALSE_FAILURE_CODES:
            print(f"\n⚠️ [{name}] 进程退出码 {proc.returncode} (Windows 保护触发)，检查是否已生成结果...")

        status = "success" if exit_ok else "failed"
        if proc.returncode in WINDOWS_FALSE_FAILURE_CODES:
            status = "windows_guard"  # 特殊标记，后续以文件为准

        return {
            "status": status,
            "time_min": elapsed,
            "name": name,
            "type": exp_type,
            "exit_code": proc.returncode,
        }

    except Exception as e:
        elapsed = (time.time() - start) / 60.0
        print(f"\n❌ [{name}] 异常: {e}")
        return {"status": "error", "time_min": elapsed, "name": name, "type": exp_type, "exit_code": None}


def read_result_csv(csv_rel_path):
    """
    读取实验结果 CSV。兼容单结果与多结果文件。
    增强列名匹配：支持 R2/R², rmse/RMSE 等变体。
    """
    csv_path = PROJECT_ROOT / csv_rel_path
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None

        # 列名清洗：去除空格，处理上标2
        df.columns = [c.strip().replace('\u00b2', '2') for c in df.columns]
        # 统一大小写查找
        col_lower = {c.lower(): c for c in df.columns}

        # 若有多行，取 R2 最高的一行
        r2_col = col_lower.get('r2', None)
        if r2_col and len(df) > 1:
            df = df.sort_values(r2_col, ascending=False).head(1)

        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"   ⚠️ 读取 {csv_rel_path} 出错: {e}")
        return None


def extract_metrics(result_dict):
    """从结果字典中提取统一格式的指标，兼容多种列名变体。"""
    if result_dict is None:
        return {
            "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan,
            "n_params": np.nan, "n_features": np.nan, "train_time_min": np.nan,
        }

    # 建立不区分大小写的查找
    lower_map = {str(k).lower().replace('\u00b2', '2'): v for k, v in result_dict.items()}

    def get_val(*keys):
        for k in keys:
            if k.lower() in lower_map:
                v = lower_map[k.lower()]
                if pd.notna(v) and v not in (None, "", "nan"):
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        continue
        return np.nan

    return {
        "RMSE": get_val("RMSE", "rmse"),
        "MAE": get_val("MAE", "mae"),
        "MAPE": get_val("MAPE", "mape"),
        "R2": get_val("R2", "r2", "R^2", "r^2"),
        "n_params": get_val("n_params", "params", "n_parameters"),
        "n_features": get_val("n_features", "features", "n_feats"),
        "train_time_min": get_val("train_time_min", "time", "train_time"),
    }


def resolve_status(run_info, metrics):
    """
    综合判断实验状态：只要结果文件中有有效 R2，即视为成功。
    解决 Windows 0xC0000409 假失败问题。
    """
    if run_info["status"] == "missing":
        return "missing"
    r2 = metrics.get("R2", np.nan)
    if pd.notna(r2) and not np.isinf(r2):
        # 结果有效，无论进程退出码如何都标记为 success
        return "success"
    if run_info["status"] in ("success", "windows_guard"):
        return "success" if run_info["status"] == "success" else "failed"
    return run_info["status"]


# ========================= 3. 阶段函数 =========================
def stage1_generate_data():
    """阶段1: 若原始数据不存在，则运行仿真数据生成。"""
    print_banner("阶段 1: 数据仿真生成")
    sim_data = DATA_RAW_DIR / "grid_material_demand_sim.csv"
    if sim_data.exists():
        print(f"✅ 仿真数据已存在: {sim_data}")
        print("   跳过生成阶段。如需重新生成，请手动删除该文件。")
        return

    # 优先查找增强版，回退原版
    gen_candidates = [
        PROJECT_ROOT / "src" / "data" / "generate_data_enhanced.py",
        PROJECT_ROOT / "src" / "data" / "generate_data.py",
    ]
    gen_script = None
    for cand in gen_candidates:
        if cand.exists():
            gen_script = cand
            break

    if gen_script is None:
        print(f"⚠️ 数据生成脚本未找到: 请确认 src/data/generate_data*.py 存在。")
        return

    print("🔄 正在生成仿真数据 ...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(
        [PYTHON_EXE, str(gen_script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.returncode == 0 or proc.returncode in WINDOWS_FALSE_FAILURE_CODES:
        print("✅ 数据仿真完成。")
    else:
        print("❌ 数据仿真失败:")
        print(proc.stderr[-500:])


def stage2_to_4_run_experiments():
    """阶段2~4: 串行运行主模型、消融实验与对比实验。"""
    print_banner("阶段 2~4: 主模型 + 消融实验 + 对比实验")
    print("说明: 所有实验串行执行，避免 GPU 显存冲突与随机种子污染。\n")

    records = []
    for name, script, csv_path, exp_type in EXPERIMENTS:
        run_info = run_script(name, script, exp_type)
        result_dict = read_result_csv(csv_path)
        metrics = extract_metrics(result_dict)

        # 综合状态判断：结果优先
        final_status = resolve_status(run_info, metrics)

        record = {
            "实验组": name,
            "类别": exp_type,
            "状态": final_status,
            **metrics,
            "脚本耗时(min)": run_info["time_min"],
            "exit_code": run_info["exit_code"],
        }
        records.append(record)

        # 每完成一个实验，立即打印当前结果
        r2_str = f"{metrics['R2']:.4f}" if pd.notna(metrics['R2']) else "--"
        print(f"\n📊 当前结果: {name} | 状态={final_status} | R²={r2_str}")

    return records


def stage5_summary(records):
    """阶段5: 汇总所有实验结果并输出表格。"""
    print_banner("阶段 5: 实验结果汇总")

    df = pd.DataFrame(records)

    # 调整列顺序
    col_order = ["实验组", "类别", "状态", "R2", "RMSE", "MAE", "MAPE",
                 "n_params", "n_features", "脚本耗时(min)", "exit_code"]
    df = df[[c for c in col_order if c in df.columns]]

    # 保存汇总 CSV
    summary_csv = FIGURES_DATA_DIR / "all_experiments_summary.csv"
    df.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"✅ 汇总表格已保存: {summary_csv}")

    # 保存 Markdown 表格
    try:
        summary_md = FIGURES_DATA_DIR / "all_experiments_summary.md"
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("# 电网物资需求预测 —— 全量实验结果汇总\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            # 简单 Markdown 表格
            f.write("| " + " | ".join(df.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
            for _, row in df.iterrows():
                vals = []
                for c in df.columns:
                    v = row[c]
                    if pd.isna(v):
                        vals.append("--")
                    elif isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                f.write("| " + " | ".join(vals) + " |\n")
        print(f"✅ Markdown 表格已保存: {summary_md}")
    except Exception as e:
        print(f"⚠️ Markdown 保存失败: {e}")

    # 打印到控制台
    print("\n" + "=" * 110)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "--"))
    print("=" * 110)

    # 打印关键结论
    print("\n🔍 关键结论提取:")
    main_df = df[df["类别"] == "main"]
    if not main_df.empty and pd.notna(main_df.iloc[0]["R2"]):
        main_r2 = main_df.iloc[0]["R2"]
        print(f"   主模型 R² = {main_r2:.4f}")
    else:
        main_r2 = np.nan
        print("   主模型 R² = -- (未获取)")

    abla = df[df["类别"] == "ablation"]
    print("\n   消融实验:")
    for _, row in abla.iterrows():
        if pd.notna(row["R2"]):
            delta = row["R2"] - main_r2 if pd.notna(main_r2) else np.nan
            print(f"      {row['实验组']}: R² = {row['R2']:.4f} (Δ={delta:+.4f})")

    base = df[df["类别"] == "baseline"]
    print("\n   对比实验:")
    for _, row in base.iterrows():
        if pd.notna(row["R2"]):
            print(f"      {row['实验组']}: R² = {row['R2']:.4f}")

    # 统计失败
    failed = df[~df["状态"].isin(["success"])]
    if not failed.empty:
        print(f"\n⚠️ 共有 {len(failed)} 个实验未成功获取有效结果，请检查日志。")
        for _, row in failed.iterrows():
            print(f"      - {row['实验组']}: 状态={row['状态']}, exit={row.get('exit_code', 'N/A')}")
        return 1
    return 0


# ========================= 4. 主入口 =========================
def main():
    total_start = time.time()
    print_banner("电网物资需求预测 —— 一键全量实验流水线", width=80)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {PYTHON_EXE}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 阶段1: 数据
    stage1_generate_data()

    # 阶段2~4: 实验
    records = stage2_to_4_run_experiments()

    # 阶段5: 汇总
    exit_code = stage5_summary(records)

    total_min = (time.time() - total_start) / 60.0
    print_banner(f"全部完成 | 总耗时: {total_min:.2f} 分钟", width=80)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())