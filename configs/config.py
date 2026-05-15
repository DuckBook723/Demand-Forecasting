# config/config.py
import os

# ======================= 1. 项目根目录 =======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================== 2. 数据文件名 =======================

# 2.1 原始模拟数据
SIM_DATASET_FILENAME = 'grid_material_demand_sim.csv'
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

# ======================= 3. 数据路径 =======================

# 3.1 数据根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
# 3.2 原始数据目录
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
SIM_DATASET_PATH = os.path.join(DATA_RAW_DIR, SIM_DATASET_FILENAME)
# 3.3 处理后数据目录
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, TRAIN_FILE_NAME)
VAL_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, VAL_FILE_NAME)
TEST_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, TEST_FILE_NAME)
SCALER_FILE_PATH = os.path.join(DATA_PROCESSED_DIR, SCALER_FILE_NAME)
DEMAND_MIN_MAX_SCALER_PATH = os.path.join(DATA_PROCESSED_DIR, DEMAND_MIN_MAX_SCALER)
# 3.4 绘图目录
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
# 3.5 特征选择目录
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
FEATURE_SELECTION_FILE_PATH = os.path.join(FEATURES_DIR, FEATURE_SELECTION_FILE_NAME)
FEATURE_EVALUATION_FILE_PATH = os.path.join(FEATURES_DIR, FEATURE_EVALUATION_FILE_NAME)
# 3.6 模型文件保存目录
MODELS_DIR = os.path.join(BASE_DIR, 'models')
# 3.7 DAE 模型文件保存路径
DAE_MODEL_FILE_PATH = os.path.join(MODELS_DIR, DAE_MODEL_FILE_NAME)
# 3.8 绘图数据保存目录
DATA_FIGURES_DIR = os.path.join(FIGURES_DIR, 'data')
# 3.9 训练模型文件保存路径
MODEL_FILE_PATH = os.path.join(MODELS_DIR, MODEL_FILE_NAME)

# ======================== 4. VMD 分解参数 ========================

DEMAND_COL = 'demand'       # 需求列
VMD_ALPHA = 2500            # 惩罚因子
VMD_TAU = 0                 # 拉格朗日乘子对偶上升步长
VMD_K = 4                   # 模态数
VMD_DC = 0                  # 直流分量约束
VMD_INIT = 0                # 中心频率的初始化方式
VMD_TOL = 1e-7              # 收敛容差

# ======================== 5. 数据集划分参数 ========================

# 5.1 训练集比例
TRAIN_RATIO = 0.7
# 5.2 验证集比例
VAL_RATIO = 0.1

# ======================= 6. 特征工程参数 =======================

# 6.1 斯皮尔曼相关性初筛阈值
SPEARMAN_THRESHOLD = 0.1
# 6.2 随机森林特征重要性评估
TOP_N = 5
# 6.3 DAE 参数
NOISE_STD = 0.1
DAE_EPOCHS = 200
DAE_BATCH_SIZE = 32
DAE_LEARNING_RATE = 1e-3
DAE_HIDDEN_DIM = [8, 3]

# ======================= 7. 数据加载器参数 =======================

# 7.1 批次大小
BATCH_SIZE = 32
# 7.2 滑动窗口长度
SEQ_LEN = 12

# ======================= 8. 模型训练参数 =======================

# 8.1 训练轮次
EPOCHS = 200
# 8.2 学习率
LEARNING_RATE = 1e-3
# 8.3 TCN隐藏层维度
TCN_HIDDEN = 64
# 8.4 TCN输出维度
TCN_OUTPUT = 32
# 8.5 LSTM隐藏层数
LSTM_HIDDEN = 64
# 8.6 LSTM层数
LSTM_LAYERS = 2
# 8.7 门控融合同一纬度
UNIFIED_DIM = 128