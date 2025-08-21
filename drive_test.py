import socket
import time
import tensorflow as tf
#import keras
import numpy as np
import cv2
import json
import os
import config

# ========== 參數區 ==========
# add by bun
# 設定寬度與高度及通道資訊
input_height = config.TARGET_IMAGE_HEIGHT
input_width = config.TARGET_IMAGE_WIDTH
input_channels = config.INPUT_CHANNELS

base_dir = config.BASE_DIR
sensor_dir = config.SENSOR_DIR
model_path = config.MODEL_PATH

unity_ip = config.UNITY_IP
unity_port = config.UNITY_PORT
image_size = (input_width, input_height)

sensor_dim = config.SENSOR_INPUT_DIM
timeout_seconds = 30
check_interval = 1  # 每秒掃描一次新圖
# ============================

# 模型載入
model = tf.keras.models.load_model(model_path)

# UDP socket 建立
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 初始化
processed_images = set()
last_image_time = time.time()

def get_latest_log_files():
    # 找到所有 .log 文件，而非目錄
    logs = [f for f in os.listdir(sensor_dir) if f.endswith(".txt") and os.path.isfile(os.path.join(sensor_dir, f))]
    if not logs:
        print("⚠️ 沒有找到任何 log 文件")
        return []
    # Add by bun setting reverse=True get latest log file.
    return sorted(logs, key=lambda f: os.path.getmtime(os.path.join(sensor_dir, f)), reverse=True)
def get_latest_folder():
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if not folders:
        return None
    latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(base_dir, f)))
    print("⚠️ 讀取 " + latest_folder)
    return os.path.join(base_dir, latest_folder)

def get_corresponding_sensor_data(timestamp):
    log_files = get_latest_log_files()
    if not log_files:
        print("⚠️ 沒有找到 log 文件")
        return None
    log_file = log_files[0]  # 取得最新的log文件

    with open(os.path.join(sensor_dir, log_file), "r") as file:
        for line in file:
            parts = line.split(";")
            if len(parts) < 12:  # 確保行資料格式正確
                continue
            log_timestamp = parts[0]
            if timestamp in log_timestamp:  # 根據時間戳匹配圖像
                try:
                    #處理 Rot，並轉為浮點數列表
                    rot = parts[4].split(":")[1].strip().strip('()').split(',')  # 去掉 "Rot: " 和括號
                    sensor_data = [float(val) for val in rot]  # 將每個數值轉為 float

                    # 處理 AngularVelocity，並轉為浮點數列表
                    angular_velocity = parts[6].split(":")[1].strip().strip('()').split(',')  # 去掉 "AngularVelocity: " 和括號
                    sensor_data += [float(val) for val in angular_velocity]  # 將每個數值轉為 float

                    sensor_data += [
                        float(parts[5].split(":")[1].strip()),  # Velocity
                        float(parts[7].split(":")[1].strip()),  # RollAngle
                        float(parts[8].split(":")[1].strip()),  # DistanceFront
                        float(parts[9].split(":")[1].strip()),  # DistanceRear
                        float(parts[10].split(":")[1].strip()),  # DistanceLeft
                        float(parts[11].split(":")[1].strip())   # DistanceRight
                    ]
                    return sensor_data
                except ValueError as e:
                    print(f"⚠️ 數據解析錯誤: {e}")
                    return None
    return None


print("🌀 開始讀取資料夾圖片...")

# ========================== 避撞邏輯 =======================
def apply_collision_avoidance(dist_front, dist_left, dist_right, steering, throttle):
    # 如果前方距離小於 3.0 公尺，停止加速並進行避撞
    # if dist_front < 3.0:
    #     print("🛑 前方過近，停止加速 + 嘗試偏轉")
    #     throttle = 0.0
    #     steering = 0.5 if dist_left > dist_right else -0.5  # 向較寬的方向轉
    if dist_left < 1.5:
        print("↪️ 太靠左，強制右偏")
        steering = min(steering + 0.8, 1.0)  # 強制右偏
    elif dist_right < 1.5:
        print("↩️ 太靠右，強制左偏")
        steering = max(steering - 0.8, -1.0)  # 強制左偏
    return steering, throttle
# ============================================================


while True:
    try:
        folder_path = get_latest_folder()
        if not folder_path:
            print("⚠️ 找不到資料夾")
            break
        
        image_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().startswith("screenshot_") and f.lower().endswith(".jpg")]
        )

        new_images = [f for f in image_files if f not in processed_images]

        if new_images:

            for filename in new_images:
                # Modify by bun to check sensor data
                timestamp = filename.split("_")[1].split(".jpg")[0]  # 從圖片檔名提取時間戳
                # timestamp = filename.split("_")[1].split(".")[0]  # 從圖片檔名提取時間戳
                sensor_data = get_corresponding_sensor_data(timestamp)

                full_path = os.path.join(folder_path, filename)
                if sensor_data:
                    print(f"📸 正在處理圖片: {filename}，對應的感測器數據: {sensor_data}")
                    img = cv2.imread(os.path.join(folder_path, filename))
                    if img is None:
                        print("❌ 圖像讀取失敗")
                        continue

                    # 預處理
                    img = cv2.resize(img, image_size)
                    img = img.astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=0)

                    # Add by bun to 正規化數據
                    """ 
                    # 後續可嘗試調整
                    normalized_sensor_data = [
                        config.norm_value(sensor_data[0], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0),
                        config.norm_value(sensor_data[1], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0),
                        config.norm_value(sensor_data[2], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0),
                        config.norm_value(sensor_data[3], config.SENSOR_RANGES['angular_velocity_x']['min'], config.SENSOR_RANGES['angular_velocity_x']['max'], target_min=-1.0, target_max=1.0),
                        config.norm_value(sensor_data[4], config.SENSOR_RANGES['angular_velocity_y']['min'], config.SENSOR_RANGES['angular_velocity_y']['max'], target_min=-1.0, target_max=1.0),
                        config.norm_value(sensor_data[5], config.SENSOR_RANGES['angular_velocity_z']['min'], config.SENSOR_RANGES['angular_velocity_z']['max'], target_min=-1.0, target_max=1.0),
                        config.norm_value(sensor_data[6], config.SENSOR_RANGES['speed']['min'], config.SENSOR_RANGES['speed']['max']),
                        config.norm_value(sensor_data[7], config.SENSOR_RANGES['rollAngle']['min'], config.SENSOR_RANGES['rollAngle']['max'], target_min=-1.0, target_max=1.0),
                        config.norm_value(sensor_data[8], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                        config.norm_value(sensor_data[9], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                        config.norm_value(sensor_data[10], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                        config.norm_value(sensor_data[11], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
                    ]
                    sensor_input = np.array(normalized_sensor_data, dtype=np.float32)
                    """
                    normalized_rotation_x = config.norm_value(sensor_data[0], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'])
                    normalized_rotation_y = config.norm_value(sensor_data[1], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'])
                    normalized_rotation_z = config.norm_value(sensor_data[2], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'])
                    normalized_angular_velocity_x = config.norm_value(sensor_data[3], config.SENSOR_RANGES['angular_velocity_x']['min'], config.SENSOR_RANGES['angular_velocity_x']['max'], target_min=-1.0, target_max=1.0)
                    normalized_angular_velocity_y = config.norm_value(sensor_data[4], config.SENSOR_RANGES['angular_velocity_y']['min'], config.SENSOR_RANGES['angular_velocity_y']['max'], target_min=-1.0, target_max=1.0)
                    normalized_angular_velocity_z = config.norm_value(sensor_data[5], config.SENSOR_RANGES['angular_velocity_z']['min'], config.SENSOR_RANGES['angular_velocity_z']['max'], target_min=-1.0, target_max=1.0)
                    normalized_speed = config.norm_value(sensor_data[6], config.SENSOR_RANGES['speed']['min'], config.SENSOR_RANGES['speed']['max'])
                    normalized_rollAngle = config.norm_value(sensor_data[7], config.SENSOR_RANGES['rollAngle']['min'], config.SENSOR_RANGES['rollAngle']['max'], target_min=-1.0, target_max=1.0)
                    normalized_distFront = config.norm_value(sensor_data[8], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
                    normalized_distRear = config.norm_value(sensor_data[9], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
                    normalized_distLeft = config.norm_value(sensor_data[10], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
                    normalized_distRight = config.norm_value(sensor_data[11], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])

                    # 感測器資料（請改成實際資料）
                    sensor_input = np.array([
                        normalized_rotation_x, normalized_rotation_y, normalized_rotation_z,
                        normalized_angular_velocity_x, normalized_angular_velocity_y, normalized_angular_velocity_z,
                        normalized_speed, normalized_rollAngle,
                        normalized_distFront, normalized_distRear, normalized_distLeft, normalized_distRight
                    ], dtype=np.float32)
                    sensor_input = np.expand_dims(sensor_input, axis=0) # 添加批次維度

                    # 預測
                    # 專屬註解：現在 model.predict 回傳一個包含兩個元素的列表，每個元素是一個 numpy 陣列
                    # prediction[0] 是轉向預測, prediction[1] 是油門預測
                    predictions = model.predict([img, sensor_input])
                    predicted_angle_normalized = predictions[0][0][0]
                    predicted_throttle_normalized = predictions[1][0][0]

                    # 預測
                    # Modify by bun to change output 
                    #prediction = model.predict([img, sensor_input])[0]
                    #predicted_angle_normalized = prediction[0]
                    #predicted_throttle_normalized = prediction[1]
                    
                    # 恢復原始數據範圍
                    steering = config.denorm_value(predicted_angle_normalized, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'], target_min=-1.0, target_max=1.0)
                    throttle = config.denorm_value(predicted_throttle_normalized, config.LABEL_RANGES['throttle']['min'], config.LABEL_RANGES['throttle']['max'], target_min=0.0, target_max=1.0)

                    # 感測器距離（假設的測距資料）
                    # Modify by bun change to correct data 
                    """
                    dist_front = sensor_data[4] # 假設前方距離 5m
                    dist_left = sensor_data[6]   # 假設左邊距離 2m
                    dist_right = sensor_data[7]  # 假設右邊距離 1m
                    """
                    dist_front = sensor_data[8] # 假設前方距離 5m
                    dist_left = sensor_data[10]   # 假設左邊距離 2m
                    dist_right = sensor_data[11]  # 假設右邊距離 1m

                    # Mark by test for bun 
                    # 應用避撞邏輯
                    # steering, throttle = apply_collision_avoidance(dist_front, dist_left, dist_right, steering, throttle)
                    
                    # 傳送
                    control_dict = {"steering": float(steering), "throttle": float(throttle)}
                    control_json = json.dumps(control_dict)

                    # 傳送
                    # Modify by bun to try float
                    #control_dict = {"steering": steering, "throttle": throttle}
                    control_dict = {"steering": float(steering), "throttle": float(throttle)}
                    control_json = json.dumps(control_dict)

                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                            sock.connect((unity_ip, unity_port))
                            sock.sendall(control_json.encode())
                            print(f"📤 傳送控制：{control_json}")

                            last_image_time = time.time()  # 重設超時計時器，Move by bun

                            processed_images.add(filename)
                    except Exception as neterr:
                        print("❌ 傳送失敗:", neterr)
                # Add by bun try to check no pic and pass.
                else:
                    print(f"⚠️ 找不到 {timestamp} 對應的感測器數據，跳過此圖片...")
                    processed_images.add(filename)

            else:
                # 沒有新圖片，檢查是否超時
                if time.time() - last_image_time > timeout_seconds:
                    print("⏱️ 30 秒內無新圖，結束")
                    break
                else:
                    print("⌛ 等待新圖片中...")
                    time.sleep(check_interval)

    except Exception as e:
        print("⚠️ 發生錯誤：", e)
        break
