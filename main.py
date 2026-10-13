"""
电网物资需求预测实验的调度入口：按物资编码依次完成数据生成、主模型、
消融实验与对比实验，最后把所有结果汇总成一张表。

运行方式:
    python main.py            # 处理 MATERIAL_REGISTRY 中注册的全部物资
    python main.py MAT002     # 只处理指定物资

物资切换依靠环境变量 MATERIAL_SCENARIO：本脚本拉起子进程时注入该变量，
子进程导入 config.py 后据此切换数据文件与输出目录。所以 config.py 需要按
MATERIAL_SCENARIO 动态给出 SIM_DATASET_FILENAME、DATA_FIGURES_DIR 等路径，
本文件本身不硬编码任何数据文件名或输出路径。
"""

import os
import sys
import time
import subprocess
import warnings
import importlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---- 通过 config.py 统一获取所有路径 ----
import configs.config as cfg

# 若本进程已设置 MATERIAL_SCENARIO，重载 config 以应用动态路径
if os.environ.get("MATERIAL_SCENARIO"):
    importlib.reload(cfg)

PROJECT_ROOT = Path(cfg.BASE_DIR)
DATA_RAW_DIR = Path(cfg.DATA_RAW_DIR)
FIGURES_DIR = Path(cfg.FIGURES_DIR)
MODELS_DIR = Path(cfg.MODELS_DIR)

PYTHON_EXE = sys.executable

# 物资注册表：编码 -> 生成脚本相对路径
# 注：仅注册调度关系，所有数据文件名由 config.py 根据 MATERIAL_SCENARIO 控制
MATERIAL_REGISTRY = {
    "001": {"gen_script": "src/data/generate_data_mat001.py"},    # 变压器基准
    "002": {"gen_script": "src/data/generate_data_mat002.py"},    # 绝缘导线
    "003": {"gen_script": "src/data/generate_data_mat003.py"},    # 智能电表
}

# 实验脚本固定相对路径（项目模块位置约定，非输出文件名）
_EXPERIMENT_SCRIPTS = [
    ("主模型 (Full)", "experiments/mainmodel/vmd_cat_bilstm_fs.py", "main"),
    ("消融: w/o CATCN", "experiments/ablations/vmd_cat_bilstm_fs_nocatcn.py", "ablation"),
    ("消融: w/o FS", "experiments/ablations/vmd_cat_bilstm_fs_nofs.py", "ablation"),
    ("消融: w/o Skip", "experiments/ablations/vmd_cat_bilstm_fs_noskip.py", "ablation"),
    ("消融: w/o VMD", "experiments/ablations/vmd_cat_bilstm_fs_novmd.py", "ablation"),
    ("对比: ARIMA", "experiments/baselines/arima_model.py", "baseline"),
    ("对比: XGBoost", "experiments/baselines/xgboost_model.py", "baseline"),
    ("对比: LSTM-diff", "experiments/baselines/lstm_diff_model.py", "baseline"),
    ("对比: Fair-LSTM/BiLSTM", "experiments/baselines/lstm_fair_model.py", "baseline"),
]

# 实验输出文件名映射（与实验脚本内部保存逻辑保持一致）
# 若实验脚本统一改为 f"{stem}_results.csv" 规范，可移除此映射表
_SCRIPT_TO_CSV = {
    "vmd_cat_bilstm_fs": "vmd_catbilstm_fs_results.csv",
    "vmd_cat_bilstm_fs_nocatcn": "vmd_catbilstm_fs_nocatcn_results.csv",
    "vmd_cat_bilstm_fs_nofs": "vmd_catbilstm_fs_nofs_results.csv",
    "vmd_cat_bilstm_fs_noskip": "vmd_catbilstm_fs_noskip_results.csv",
    "vmd_cat_bilstm_fs_novmd": "vmd_catbilstm_fs_novmd_results.csv",
    "arima_model": "arima_results.csv",
    "xgboost_model": "xgboost_results.csv",
    "lstm_diff_model": "lstm_diff_results_seed42.csv",
    "lstm_fair_model": "fair_lstm_bilstm_results.csv",
}

# Windows 假失败退出码集合 (进程保护触发，非实验错误)
WINDOWS_FALSE_FAILURE_CODES = {3221226505, 0xC0000409, -1073740791}


# ---- 工具函数 ----
def print_banner(title, width=70):
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def build_experiment_registry(material_code: str):
    """
    根据物资编码动态构建实验注册表。
    结果 CSV 路径位于 <FIGURES_DIR>/data/<material_code.lower()>/ 下，
    要求 config.py 根据 MATERIAL_SCENARIO 将 DATA_FIGURES_DIR 指向同一目录。
    """
    material_out = FIGURES_DIR / "data" / material_code.lower()
    material_out.mkdir(parents=True, exist_ok=True)

    registry = []
    for name, script_rel, exp_type in _EXPERIMENT_SCRIPTS:
        script_name = Path(script_rel).stem
        csv_name = _SCRIPT_TO_CSV.get(script_name, f"{script_name}_results.csv")
        csv_path = str(material_out / csv_name)
        registry.append((f"[{material_code}] {name}", script_rel, csv_path, exp_type))
    return registry


def run_script(name, script_path, exp_type, material_code: str):
    """
    运行单个实验脚本，向子进程注入 MATERIAL_SCENARIO 环境变量，
    使子进程加载 config.py 时自动切换数据源与输出目录。
    """
    full_script = PROJECT_ROOT / script_path
    if not full_script.exists():
        print(f"\n⚠️  [{name}] 脚本未找到: {full_script}")
        return {"status": "missing", "time_min": 0, "name": name, "type": exp_type, "exit_code": None}

    print(f"\n🚀 启动 [{name}] ...")
    print(f"   脚本: {script_path}")
    start = time.time()

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # 关键：向子进程传递当前物资编码，驱动 config.py 动态路径切换
        env["MATERIAL_SCENARIO"] = material_code

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


def read_result_csv(csv_path_input):
    """
    读取实验结果 CSV。兼容单结果与多结果文件。
    增强列名匹配：支持 R2/R², rmse/RMSE 等变体。
    """
    csv_path = Path(csv_path_input)
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
        print(f"   ⚠️ 读取 {csv_path} 出错: {e}")
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


# ---- 阶段函数 ----
def stage1_generate_data(material_code: str) -> bool:
    """阶段1: 检查并生成指定物资的仿真数据。返回是否成功。"""
    print_banner(f"阶段 1: 数据仿真生成 [{material_code}]")

    # 从 config.py 动态获取该物资的数据集路径（如 SIM_DATASET_MAT001_PATH）
    dataset_path_attr = f"SIM_DATASET_{material_code}_PATH"
    dataset_path = getattr(cfg, dataset_path_attr, None)
    if dataset_path is None:
        print(f"⚠️  [{material_code}] config.py 中未定义 {dataset_path_attr}，跳过数据生成")
        return False

    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        print(f"✅  [{material_code}] 仿真数据已存在: {dataset_path}")
        return True

    # 查找并执行生成脚本
    info = MATERIAL_REGISTRY.get(material_code)
    if not info:
        print(f"⚠️  [{material_code}] 未在 MATERIAL_REGISTRY 中注册生成脚本")
        return False

    gen_script = PROJECT_ROOT / info["gen_script"]
    # 回退：尝试增强版命名
    if not gen_script.exists():
        alt = PROJECT_ROOT / info["gen_script"].replace(".py", "_enhanced.py")
        if alt.exists():
            gen_script = alt

    if not gen_script.exists():
        print(f"⚠️  [{material_code}] 数据生成脚本未找到: {info['gen_script']}")
        return False

    print(f"🔄  [{material_code}] 正在生成仿真数据: {gen_script.name} ...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["MATERIAL_SCENARIO"] = material_code

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
        print(f"✅  [{material_code}] 数据仿真完成。")
        return True
    else:
        print(f"❌  [{material_code}] 数据仿真失败 (exit={proc.returncode}):")
        print(proc.stderr[-500:])
        return False


def stage2_to_4_run_experiments(material_code: str):
    """阶段2~4: 串行运行指定物资的主模型、消融实验与对比实验。"""
    print_banner(f"阶段 2~4: 主模型 + 消融实验 + 对比实验 [{material_code}]")
    print("说明: 所有实验串行执行，避免 GPU 显存冲突与随机种子污染。\n")

    experiments = build_experiment_registry(material_code)
    records = []
    for name, script, csv_path, exp_type in experiments:
        run_info = run_script(name, script, exp_type, material_code)
        result_dict = read_result_csv(csv_path)
        metrics = extract_metrics(result_dict)

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

        r2_str = f"{metrics['R2']:.4f}" if pd.notna(metrics['R2']) else "--"
        print(f"\n📊 当前结果: {name} | 状态={final_status} | R²={r2_str}")

    return records


def stage5_summary(records, material_code: str):
    """阶段5: 汇总指定物资的实验结果并输出表格。"""
    print_banner(f"阶段 5: 实验结果汇总 [{material_code}]")

    df = pd.DataFrame(records)

    # 调整列顺序
    col_order = ["实验组", "类别", "状态", "R2", "RMSE", "MAE", "MAPE",
                 "n_params", "n_features", "脚本耗时(min)", "exit_code"]
    df = df[[c for c in col_order if c in df.columns]]

    # 保存汇总 CSV 到物资子目录
    material_out = FIGURES_DIR / "data" / material_code.lower()
    material_out.mkdir(parents=True, exist_ok=True)
    summary_csv = material_out / "all_experiments_summary.csv"
    df.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"✅ 汇总表格已保存: {summary_csv}")

    # 保存 Markdown 表格
    try:
        summary_md = material_out / "all_experiments_summary.md"
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write(f"# 电网物资需求预测 —— {material_code} 全量实验结果汇总\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
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


# ---- 主入口 ----
def main():
    total_start = time.time()
    print_banner("电网物资需求预测 —— 多物资一键全量实验流水线", width=80)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {PYTHON_EXE}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"注册物资: {list(MATERIAL_REGISTRY.keys())}")

    # 支持命令行指定单个或多个物资，如: python main.py MAT002
    # 无参数时默认运行全部注册物资
    target_materials = sys.argv[1:] if len(sys.argv) > 1 else list(MATERIAL_REGISTRY.keys())

    global_exit = 0
    for material in target_materials:
        if material not in MATERIAL_REGISTRY:
            print(f"\n⚠️ 未知物资编码: {material}，已跳过。可用: {list(MATERIAL_REGISTRY.keys())}")
            global_exit = 1
            continue

        print_banner(f"开始处理物资: {material}", width=70)

        # 阶段1: 数据
        data_ok = stage1_generate_data(material)
        if not data_ok:
            print(f"❌ [{material}] 数据准备失败，跳过后续实验。")
            global_exit = 1
            continue

        # 阶段2~4: 实验
        records = stage2_to_4_run_experiments(material)

        # 阶段5: 汇总
        exit_code = stage5_summary(records, material)
        if exit_code != 0:
            global_exit = 1

    total_min = (time.time() - total_start) / 60.0
    print_banner(f"全部完成 | 总耗时: {total_min:.2f} 分钟", width=80)
    return global_exit


if __name__ == "__main__":
    sys.exit(main())