# config/config.py
import os

# ======================= 0. 动态物资切换 =======================
# 主调度脚本(main.py)通过环境变量 MATERIAL_SCENARIO 控制当前实验的物资编码
# 实验子进程导入本模块时自动切换所有路径，无需修改任何实验脚本内部逻辑
_MATERIAL_SCENARIO = os.environ.get("MATERIAL_SCENARIO", "003").upper()
_VALID_MATERIALS = {"001", "002", "003"}

if _MATERIAL_SCENARIO not in _VALID_MATERIALS:
    _MATERIAL_SCENARIO = "002"

_MATERIAL_CODE_LOWER = _MATERIAL_SCENARIO.lower()
# 例如: "mat001", "mat002"

# ======================= 1. 项目根目录 =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================== 2. 数据文件名 =======================

# 2.1 原始模拟数据（三物资固定命名，供生成脚本与主调度直接引用）
SIM_DATASET_001_FILENAME = 'grid_material_001.csv'          # MAT001 变压器
SIM_DATASET_002_FILENAME = 'grid_material_002.csv'          # MAT002 绝缘导线
SIM_DATASET_003_FILENAME = 'grid_material_003.csv'          # MAT003 智能电表

# ★ 动态切换：当前激活的物资数据集文件名 ★
# 实验脚本通过 SIM_DATASET_PATH 自动获得当前物资的数据文件
SIM_DATASET_FILENAME = f'grid_material_{_MATERIAL_CODE_LOWER}.csv'

MATERIAL_DEMAND_DIR = 'material_monthly'

# 2.2 处理后数据文件名
TRAIN_FILE_NAME = 'train.csv'
VAL_FILE_NAME = 'val.csv'
TEST_FILE_NAME = 'test.csv'
SCALER_FILE_NAME = 'scaler.pkl'
DEMAND_MIN_MAX_SCALER = 'demand_min_max_scaler.pkl'

# 2.3 特征选择文件名
FEATURE_SELECTION_FILE_NAME = 'selected_features.json'

# 2.4 特征评估数据文件名
FEATURE_EVALUATION_FILE_NAME = 'feature_evaluation.csv'

# 2.6 DAE 模型文件名
DAE_MODEL_FILE_NAME = 'dae_model.pth'

# 2.7 训练模型文件名
MODEL_FILE_NAME = 'vmd_cat_bilstm_model.pth'

# 2.8 训练记录保存文件名
TRAIN_RECORD_FILE_NAME = 'train_record.csv'

# 2.9 字体文件名
FONT_SIMHEI_FILE_NAME = 'SimHei.ttf'
FONT_SARASAMONOSC_BOLD_FILE_NAME = 'SarasaMonoSC-Bold.ttf'
FONT_SARASAMONOSC_REGULAR_FILE_NAME = 'SarasaMonoSC-Regular.ttf'
FONT_DEJAVUSANS_FILE_NAME = 'DejaVuSans.ttf'

# ======================= 3. 数据路径 =======================

# 3.1 数据根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 3.2 原始数据目录（所有物资共享同一原始目录）
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
SIM_DATASET_PATH = os.path.join(DATA_RAW_DIR, SIM_DATASET_FILENAME)
MATERIAL_DEMAND_PATH = os.path.join(DATA_RAW_DIR, MATERIAL_DEMAND_DIR)

# 静态路径：供 main.py stage1 直接引用，检查各物资原始数据是否存在
SIM_DATASET_001_PATH = os.path.join(DATA_RAW_DIR, SIM_DATASET_001_FILENAME)
SIM_DATASET_002_PATH = os.path.join(DATA_RAW_DIR, SIM_DATASET_002_FILENAME)
SIM_DATASET_003_PATH = os.path.join(DATA_RAW_DIR, SIM_DATASET_003_FILENAME)

# 3.3 处理后数据目录（★ 按物资分子目录，避免训练集/标准化参数覆盖 ★）
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed', _MATERIAL_CODE_LOWER)
TRAIN_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, TRAIN_FILE_NAME)
VAL_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, VAL_FILE_NAME)
TEST_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, TEST_FILE_NAME)
SCALER_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, SCALER_FILE_NAME)
DEMAND_MIN_MAX_SCALER_PATH = os.path.join(DATA_PROCESSED_DIR, DEMAND_MIN_MAX_SCALER)

# 3.4 绘图目录
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# 3.5 特征选择目录（★ 按物资分子目录 ★）
FEATURES_DIR = os.path.join(DATA_DIR, 'features', _MATERIAL_CODE_LOWER)
FEATURE_SELECTION_FILE_PATH = os.path.join(FEATURES_DIR, FEATURE_SELECTION_FILE_NAME)
FEATURE_EVALUATION_FILE_PATH = os.path.join(FEATURES_DIR, FEATURE_EVALUATION_FILE_NAME)

# 3.6 模型文件保存目录（★ 按物资分子目录，避免权重文件覆盖 ★）
MODELS_DIR = os.path.join(BASE_DIR, 'models', _MATERIAL_CODE_LOWER)

# 3.7 DAE 模型文件保存路径
DAE_MODEL_FILE_PATH = os.path.join(MODELS_DIR, DAE_MODEL_FILE_NAME)

# 3.8 绘图数据保存目录（★ 按物资分子目录，实验结果CSV自动隔离 ★）
DATA_FIGURES_DIR = os.path.join(FIGURES_DIR, 'data', _MATERIAL_CODE_LOWER)

# 3.9 训练模型文件保存路径
MODEL_FILE_PATH = os.path.join(MODELS_DIR, MODEL_FILE_NAME)

# 3.10 训练记录保存路径
TRAIN_RECORD_FILE_PATH = os.path.join(DATA_FIGURES_DIR, TRAIN_RECORD_FILE_NAME)

# 3.11 资源文件目录
RESOURCE_DIR = os.path.join(BASE_DIR, 'assets')

# 3.12 字体保存路径
FONT_PATH = os.path.join(RESOURCE_DIR, 'fonts')
FONT_SIMHEI_FILE_PATH = os.path.join(FONT_PATH, FONT_SIMHEI_FILE_NAME)
FONT_SARASAMONOSC_BOLD_FILE_PATH = os.path.join(FONT_PATH, FONT_SARASAMONOSC_BOLD_FILE_NAME)
FONT_SARASAMONOSC_REGULAR_FILE_PATH = os.path.join(FONT_PATH, FONT_SARASAMONOSC_REGULAR_FILE_NAME)
FONT_DEJAVUSANS_FILE_PATH = os.path.join(FONT_PATH, FONT_DEJAVUSANS_FILE_NAME)

# ★ 自动创建所有动态子目录，确保子进程写入时路径存在 ★
for _d in [DATA_PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, DATA_FIGURES_DIR]:
    os.makedirs(_d, exist_ok=True)

# ======================== 4. VMD 分解参数 ========================

DEMAND_COL = 'demand'           # 需求列
VMD_ALPHA = 500                 # 惩罚因子
VMD_TAU = 0                     # 拉格朗日乘子对偶上升步长
VMD_K = 3                       # 模态数
VMD_DC = 0                      # 直流分量约束
VMD_INIT = 0                    # 中心频率的初始化方式
VMD_TOL = 1e-7                  # 收敛容差

# ======================== 5. 数据集划分参数 ========================

# 5.1 训练集比例
TRAIN_RATIO = 0.6
# 5.2 验证集比例
VAL_RATIO = 0.2

# ======================= 6. 特征工程参数 =======================

# 6.1 斯皮尔曼相关性初筛阈值
SPEARMAN_THRESHOLD = 0.1
# 6.2 随机森林特征重要性评估
TOP_N = 7
# 6.3 DAE 参数
NOISE_STD = 0.1
DAE_EPOCHS = 500
DAE_BATCH_SIZE = 32
DAE_LEARNING_RATE = 1e-3
DAE_HIDDEN_DIM = [16, 8]

# ======================= 7. 数据加载器参数 =======================

# 7.1 批次大小
BATCH_SIZE = 16
# 7.2 滑动窗口长度
SEQ_LEN = 12

# ======================= 8. 模型训练参数 =======================

# 8.1 训练轮次
EPOCHS = 150
# 8.2 学习率
LEARNING_RATE = 1e-3
# 8.3 TCN隐藏层维度
TCN_HIDDEN = 16
# 8.4 TCN输出维度
TCN_OUTPUT = 8
# 8.5 LSTM隐藏层维度
LSTM_HIDDEN = 16
# 8.6 LSTM层数
LSTM_LAYERS = 2
# 8.7 门控融合同一纬度
UNIFIED_DIM = 16
# 8.8 L2 正则化权重衰减系数
WEIGHT_DECAY = 1e-5
# 8.9 dropout 概率
DROPOUT = 0.2
# 8.10 早停开关
EARLY_STOPPING = True
# 8.11 早停耐心值
PATIENCE = 30
# 8.12 SubNet参数共享
SHARE_SUBNET = True
# 8.13 DAE端到端训练开关
DAE_END_TO_END = True
# 8.14 DAE重构辅助损失权重
RECON_LOSS_WEIGHT = 0.1
# 8.15 随机种子
RANDOM_SEED = 42

# ======================= 9. 实验相关路径 =======================

# 9.1 实验目录
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')
# 9.2 消融实验模型保存目录（基于动态 MODELS_DIR）
ABLATION_MODELS_DIR = os.path.join(MODELS_DIR, 'ablation')
# 9.3 消融实验训练记录保存目录（基于动态 DATA_FIGURES_DIR）
ABLATION_RECORDS_DIR = os.path.join(DATA_FIGURES_DIR, 'ablation')
# 9.4 消融实验汇总结果文件路径
ABLATION_RESULTS_PATH = os.path.join(DATA_FIGURES_DIR, 'ablation_results.csv')

# ======================= 测试 =======================
if __name__ == '__main__':

    # 打印当前激活的物资场景（用于调试路径是否正确）
    print(f"当前物资场景: {_MATERIAL_SCENARIO}")
    print(f"数据集路径: {SIM_DATASET_PATH}")
    print(f"处理后目录: {DATA_PROCESSED_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"结果目录: {DATA_FIGURES_DIR}")
