# config.py
# 共用參數設定配置

# =========================================================================
# 溝通資訊參數區
# =========================================================================
UNITY_IP = "192.168.0.13" # "140.110.90.55" 192.168.0.13
UNITY_PORT = 12352

# =========================================================================
# 資料夾參數區
# =========================================================================
# Create data list sensor
TIMESTAMP_NAME = '20250728_184331'
LOG_PATH = 'C:/Users/User/source/repos/Car02/Event/Event_'+TIMESTAMP_NAME+'.txt'   # log檔路徑 
SENSOR_PATH = 'C:/Users/User/source/repos/Car02/Log/Log_'+TIMESTAMP_NAME+'.txt'   # sensor檔路徑
IMAGE_FOLDER = 'C:/Users/User/source/repos/Car02/Cam01/'+TIMESTAMP_NAME         # 照片資料夾
SAVE_DIR = './data'     # 輸出 train.txt/val.txt 的位置
# LOG_PATH # 'C:/Users/Clary Lin/Downloads/20250706Car01/20250506Car01/Car0706/Event/Event_'+TIMESTAMP_NAME+'.txt'   # log檔路徑
# SENSOR_PATH # 'C:/Users/Clary Lin/Downloads/20250706Car01/20250506Car01/Car0706/Log/Log_'+TIMESTAMP_NAME+'.txt'   # sensor檔路徑
# IMAGE_FOLDER # 'C:/Users/Clary Lin/Downloads/20250706Car01/20250506Car01/Car0706/Cam01/'+TIMESTAMP_NAME         # 照片資料夾


# Driver test
USER_NAME = 'User'
# USER_NAME = 'Clary Lin'
BASE_DIR = "C:/Users/"+USER_NAME+"/source/repos/Car02/Cam01/" # 圖檔資料夾位置
SENSOR_DIR = "C:/Users/"+USER_NAME+"/source/repos/Car02/Log/" # 感測器資訊位置
MODEL_PATH = "./checkpoints/model_epoch_001.h5" # 讀取訓練模型位置
# MODEL_SAVE_PATH = 'saved_models/my_multi_modal_cnn_model.h5' # 模型保存路徑
# BASE_DIR = "C:/Users/Clary Lin/Downloads/20250706Car01/20250506Car01/Car0706/Cam01/" # 圖檔資料夾位置
# SENSOR_DIR = "C:/Users/Clary Lin/Downloads/20250706Car01/20250506Car01/Car0706/Log/" # 感測器資訊位置

# Parse_txt_file
TRAIN_DIR_PATH = './data/train.txt' # 替換為您的 train.txt 路徑

# =========================================================================
# 圖像處理相關設定
# =========================================================================
TARGET_IMAGE_HEIGHT = 356  # 圖像目標高度
TARGET_IMAGE_WIDTH = 634   # 圖像目標寬度
TARGET_IMAGE_SIZE = (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT) # OpenCV 或 PIL 使用 (寬, 高)
INPUT_CHANNELS = 3

# =========================================================================
# 模型相關設定
# =========================================================================
BATCH_SIZE = 64 # 訓練批次大小
NUM_EPOCHS = 32 # 訓練 epoch 數量
LEARNING_RATE = 1e-4 # 學習率
TRAIN_RATIO = 0.8 # 訓練比率

# =========================================================================
# 感測器數據正規化範圍設定
# 目前是根據查到的數據做設定的。
# 嘗試在訓練和預測時都使用這些範圍進行正規化及反向操作。
# =========================================================================
SENSOR_RANGES = {
    'speed': {'min': 0.0, 'max': 100.0},
    'rotation_xyz': {'min': 0.0, 'max': 360.0}, # Rotation X, Y, Z (Pitch, Yaw, Roll)
    'angular_velocity_x': {'min': -1.0, 'max': 1.0},
    'angular_velocity_y': {'min': -1.0, 'max': 1.0},
    'angular_velocity_z': {'min': -1.0, 'max': 1.0},
    'rollAngle': {'min': -180.0, 'max': 180.0},
    'distance': {'min': 0.0, 'max': 30.0}, # Front, Rear, Left, Right Distance
}

# =========================================================================
# 輸出數據正規化範圍設定
# 目前是根據查到的數據做設定的。
# 嘗試在訓練和預測時都使用這些範圍進行正規化及反向操作。
# =========================================================================
LABEL_RANGES = {
    'angle': {'min': -30.0, 'max': 30.0},
    'throttle': {'min': 0.0, 'max': 1.0},
}

# =========================================================================
# 模型輸入維度 (感測器部分)
# 根據 SENSOR_RANGES 中定義的感測器數量計算而來:
# rotation_xyz (3) + angular_velocity_xyz (3) + speed (1) + rollAngle (1) + distance (4) = 12
# 0903 嘗試新增 車道線資訊
# =========================================================================
SENSOR_INPUT_DIM = 13

# =========================================================================
# Add by bun for test -1 to 1
# 數據正規化
# =========================================================================
def norm_value(value, original_min, original_max, target_min=0.0, target_max=1.0):
    if original_max == original_min:
        return target_min # 避免除以零
    normalized_0_1 = (value - original_min) / (original_max - original_min)
    return normalized_0_1 * (target_max - target_min) + target_min

# =========================================================================
# Add by bun for test -1 to 1
#助還原數據
# =========================================================================
def denorm_value(normalized_value, original_min, original_max, target_min=0.0, target_max=1.0):
    if target_max == target_min:
        return original_min # 避免除以零
    denormalized_0_1 = (normalized_value - target_min) / (target_max - target_min)
    return denormalized_0_1 * (original_max - original_min) + original_min
