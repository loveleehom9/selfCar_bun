# test_golden_sample.py
import tensorflow as tf
import numpy as np
import cv2
import os
import json
# 確保您的 models.py 和 config.py 在相同的目錄下
from models import create_autodrive_model
from config import config

"""
def norm_value(value, min_value, max_value, target_min=-1.0, target_max=1.0):
    # 將值從原始範圍歸一化到目標範圍。
    normalized_value = target_min + (value - min_value) * (target_max - target_min) / (max_value - min_value)
    return normalized_value

def denorm_value(normalized_value, min_value, max_value, target_min=-1.0, target_max=1.0):
    # 將值從目標範圍反歸一化回原始範圍。
    denormalized_value = min_value + (normalized_value - target_min) * (max_value - min_value) / (target_max - target_min)
    return denormalized_value


"""

def preprocess_image(img_path, target_size=(634, 356)):
    """
    預處理圖像：讀取、調整大小、正規化、轉換形狀
    注意: 您目前使用 BGR -> HSV 的顏色轉換，這與我先前的建議不同，但只要您的模型是使用 HSV 訓練的，這就是正確的。
    """
    img = cv2.imread(img_path)  # 讀取 BGR 圖像
    if img is None:
        print(f"錯誤：無法讀取圖片 {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 轉為 HSV
    img = cv2.resize(img, target_size)  # 確保輸入尺寸符合模型需求
    img = img.astype(np.float32) / 255.0  # 歸一化到 [0, 1]
    img = np.expand_dims(img, axis=0)  # 增加 batch 維度 -> (1, H, W, 3)
    return img

def load_and_normalize_sensor_data(json_path):
    """
    從 JSON 檔案中載入原始感測器數據，並根據 config.py 進行歸一化。
    """
    if not os.path.exists(json_path):
        print(f"錯誤：找不到感測器檔案 {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取原始感測器數據
    raw_sensors = np.array([
        data["Rotation"]["x"], data["Rotation"]["y"], data["Rotation"]["z"],
        data["AngularVelocity"]["x"], data["AngularVelocity"]["y"], data["AngularVelocity"]["z"],
        data["Velocity"]["x"], data["Velocity"]["y"], data["Velocity"]["z"],
        data["Distance"]["front"], data["Distance"]["left"], data["Distance"]["right"]
    ])

    # 根據 config.py 進行歸一化
    normalized_sensors = []
    # 旋轉角度 (Rotation)
    for value in raw_sensors[0:3]:
        normalized_sensors.append(config.norm_value(value, config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max']))
    # 角速度 (AngularVelocity)
    for value in raw_sensors[3:6]:
        normalized_sensors.append(config.norm_value(value, config.SENSOR_RANGES['angular_velocity_xyz']['min'], config.SENSOR_RANGES['angular_velocity_xyz']['max']))
    # 速度 (Velocity)
    for value in raw_sensors[6:9]:
        normalized_sensors.append(config.norm_value(value, config.SENSOR_RANGES['velocity_xyz']['min'], config.SENSOR_RANGES['velocity_xyz']['max']))
    # 距離 (Distance)，注意此處的歸一化目標範圍為 [0, 1]
    for value in raw_sensors[9:12]:
        normalized_sensors.append(config.norm_value(value, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'], target_min=0.0, target_max=1.0))
        
    normalized_sensors = np.array(normalized_sensors, dtype=np.float32)
    return np.expand_dims(normalized_sensors, axis=0), raw_sensors

def main():
    # 模型與圖片路徑
    model_path = './checkpoints/model_epoch_016.h5'
    
    # 請將以下路徑替換為您的「黃金樣本」路徑
    img_path = 'golden_sample.jpg'
    json_path = 'golden_sample.json'

    # 載入模型
    try:
        model = create_autodrive_model(input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), sensor_input_dim=12)
        model.load_weights(model_path)
        print("✅ 模型已載入。")
    except Exception as e:
        print(f"錯誤：無法載入模型。請確認檔案路徑 {model_path} 和 models.py 的正確性。詳細錯誤：{e}")
        return

    # 圖片預處理
    img = preprocess_image(img_path, target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    if img is None:
        return

    # 獲取並歸一化感測器數據
    sensor_data, raw_sensor_data = load_and_normalize_sensor_data(json_path)
    if sensor_data is None:
        return

    print("\n--- 樣本數據 ---")
    print(f"測試圖片: {img_path}")
    print(f"原始感測器數據: {raw_sensor_data}")
    
    # 預測
    try:
        prediction_normalized = model.predict({'image_input': img, 'sensor_input': sensor_data})[0]
        
        # 反歸一化預測值，得到真實世界的數值
        predicted_steering_normalized = prediction_normalized[0]
        predicted_throttle_normalized = prediction_normalized[1]
        
        predicted_steering = config.denorm_value(predicted_steering_normalized, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'])
        predicted_throttle = config.denorm_value(predicted_throttle_normalized, config.LABEL_RANGES['throttle']['min'], config.LABEL_RANGES['throttle']['max'], target_min=0.0, target_max=1.0)
        
        # 顯示預測結果
        print("\n--- 預測結果 ---")
        print(f"正規化轉向角: {predicted_steering_normalized:.4f}")
        print(f"反歸一化轉向角: {predicted_steering:.4f} 度")
        print(f"正規化油門: {predicted_throttle_normalized:.4f}")
        print(f"反歸一化油門: {predicted_throttle:.4f}")
    
    except Exception as e:
        print(f"模型預測失敗：{e}")


if __name__ == '__main__':
    main()
