import tensorflow as tf
import numpy as np
import cv2
import os
from models import create_autodrive_model  # 匯入你自己定義的模型

def preprocess_image(img_path, target_size=(634, 356)):
    """
    預處理圖像：讀取、調整大小、正規化、轉換形狀
    """
    img = cv2.imread(img_path)  # 讀取 BGR 圖像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 轉為 HSV
    img = cv2.resize(img, target_size)  # 確保輸入尺寸符合模型需求
    img = img.astype(np.float32) / 255.0  # 歸一化到 [0, 1]
    img = np.expand_dims(img, axis=0)  # 增加 batch 維度 -> (1, H, W, 3)
    return img

def get_sensor_data():
    """
    返回一個示範的感測器數據（在實際情況中，你應該將此數據替換為真實的感測器輸入）
    """
    # 假設感測器數據的結構（12 維）
    # sensor_data = np.random.rand(12)  # 隨機生成一組數據
    # sensor_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.346024e-06, 0.0, 30.0, 0.62, 4.5, 4.5]
    sensor_data = [ 9.9849999e-01,9.9938887e-01,-1.0000000e+00,2.0000000e-02,\
        0.0000000e+00,0.0000000e+00,2.0661560e-01,-6.1111111e-04,5.9166664e-01,\
      1.0000000e+00,1.4766666e-01,1.5233333e-01]
    return np.expand_dims(sensor_data, axis=0)  # 增加 batch 維度 -> (1, 12)

# add by bun
def parse_sensor_data_from_log(log_path, target_timestamp):
    """
    從日誌檔案中解析並提取特定時間戳的 12 個感測器輸入數據。
    Returns a numpy array of shape (1, 12) or None if not found.
    """
    if not os.path.exists(log_path):
        print(f"❌ 錯誤：找不到感測器數據檔案 {log_path}")
        return None

    try:
        with open(log_path, 'r') as f:
            for line in f:
                # 跳過標頭和空行
                if line.startswith("StartDateTime") or line.startswith("Timestamp") or not line.strip():
                    continue
                
                parts = line.strip().split(';')
                if len(parts) < 3:
                    continue

                # 格式化時間戳以匹配檔案名
                log_timestamp = f"{float(parts[0].strip()):.2f}"
                
                # 找到對應的行
                if log_timestamp == target_timestamp:
                    
                    # 初始化感測器數值列表
                    sensor_values = []
                    
                    # 提取 12 個感測器數據作為模型輸入
                    # 提取順序：
                    # 1. Velocity (1 value)
                    # 2. Rot (3 values: x, y, z)
                    # 3. AngularVelocity (3 values: x, y, z)
                    # 4. rollAngle (1 value)
                    # 5. DistanceFront (1 value)
                    # 6. DistanceRear (1 value)
                    # 7. DistanceLeft (1 value)
                    # 8. DistanceRight (1 value)
                    for part in parts:
                        part = part.strip()
                        if part.startswith("Velocity:"):
                            value = float(part.replace("Velocity:", "").strip())
                            sensor_values.append(value)
                        elif part.startswith("Rot:"):
                            values = part.replace("Rot: (", "").replace(")", "").strip().split(', ')
                            sensor_values.extend([float(v) for v in values])
                        elif part.startswith("AngularVelocity:"):
                            values = part.replace("AngularVelocity: (", "").replace(")", "").strip().split(', ')
                            sensor_values.extend([float(v) for v in values])
                        elif part.startswith("rollAngle:"):
                            value = float(part.replace("rollAngle:", "").strip())
                            sensor_values.append(value)
                        elif part.startswith("DistanceFront:"):
                            value = float(part.replace("DistanceFront:", "").strip())
                            sensor_values.append(value)
                        elif part.startswith("DistanceRear:"):
                            value = float(part.replace("DistanceRear:", "").strip())
                            sensor_values.append(value)
                        elif part.startswith("DistanceLeft:"):
                            value = float(part.replace("DistanceLeft:", "").strip())
                            sensor_values.append(value)
                        elif part.startswith("DistanceRight:"):
                            value = float(part.replace("DistanceRight:", "").strip())
                            sensor_values.append(value)

                    # 檢查是否所有 12 個感測器值都找到了
                    if len(sensor_values) == 12:
                        print(f"✅ 找到時間戳 {target_timestamp} 的感測器輸入數據：{sensor_values}")
                        return np.array(sensor_values, dtype=np.float32).reshape(1, -1)
                    else:
                        print(f"⚠️ 警告：時間戳 {target_timestamp} 的感測器數據長度不正確 ({len(sensor_values)}/12)。")
                        return None
        
        print(f"❌ 錯誤：在日誌檔案中找不到時間戳為 {target_timestamp} 的數據。")
        return None
    except Exception as e:
        print(f"❌ 錯誤：解析日誌檔案時發生問題。詳細錯誤訊息：{e}")
        return None

# add by bun
def extract_timestamp_from_filename(img_path):
    """
    從圖像檔名中提取時間戳。
    例如：'Screenshot_0.00.jpg' -> '0.00'
    """
    filename = os.path.basename(img_path)
    if '_' in filename and '.' in filename:
        timestamp_str = filename.split('_')[-1].split('.')[0]
        try:
            return f"{float(timestamp_str):.2f}"
        except ValueError:
            return None
    return None

def main():
    # 模型與圖片路徑
    model_path = './checkpoints/model_epoch_008.h5'  # 訓練好的模型
    img_path = 'C:/Users/User/source/repos/Car02/Cam01/20250801_115515/Screenshot_0.00.jpg'  # 單張測試圖片
    SENSOR_PATH = 'C:/Users/User/source/repos/Car02/Log/Log_20250801_115515.txt'   # sensor檔路徑

    # 載入模型
    #model = create_autodrive_model(input_shape=(712, 1267, 3), sensor_input_dim=12)
    model = create_autodrive_model(input_shape=(356, 634, 3), sensor_input_dim=12)
    model.load_weights(model_path)
    print("✅ 模型已載入")

    # 圖片預處理
    img = preprocess_image(img_path)

    # 獲取感測器數據（此為示範，應該來自你的感測器數據）
    # sensor_data = get_sensor_data()

    # 提取檔名中的時間戳
    target_timestamp = extract_timestamp_from_filename(img_path)
    if target_timestamp is None:
        print(f"❌ 錯誤：無法從圖片檔名 {img_path} 中提取時間戳。")
        return

    # 根據時間戳從日誌檔案中獲取對應的感測器數據（模型輸入）
    sensor_data = parse_sensor_data_from_log(SENSOR_PATH, target_timestamp)
    if sensor_data is None:
        return

    # 進行預測，模型會輸出轉向角和油門大小
    # The model will predict the steering angle and throttle
    predictions = model.predict({'image_input': img, 'sensor_input': sensor_data}) # 提供兩個輸入
    
    # 模型的兩個輸出分別對應轉向角和油門大小
    # The two outputs of the model correspond to steering angle and throttle
    #angle = float(prediction[0])
    #throttle = float(prediction[1])
    # 檢查預測輸出的長度以避免索引錯誤
    # Check the length of the prediction output to avoid index errors
    if len(predictions) == 2:
        # 正確地從清單中提取每個輸出層的預測值
        # Correctly extract the prediction value from each output layer's array
        angle = float(predictions[0][0][0])
        throttle = float(predictions[1][0][0])
        print(f"🚗 預測結果 — 轉向角: {angle:.4f}; 油門: {throttle:.4f}")
    else:
        print(f"❌ 錯誤：預測結果數量不正確。預期有 2 個輸出，但實際有 {len(predictions)} 個。")


    # 顯示預測結果
    print(f"🚗 預測結果 — 轉向角: {angle:.4f}; 油門: {throttle:.4f}")

    """
    # 預測
    prediction = model.predict({'image_input': img, 'sensor_input': sensor_data})[0]  # 提供兩個輸入
    angle = float(prediction[0])
    throttle = float(prediction[1])

    # 顯示預測結果
    print(f"🚗 預測角度: {angle:.4f}; 油門: {throttle:.4f}")
    """

if __name__ == '__main__':
    main()
