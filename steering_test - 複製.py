import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import random

# =========================================================================
# 模擬 config 模組和設定 (已更新為真實路徑)
# =========================================================================
class MockConfig:
    # 圖像目標高度與寬度
    TARGET_IMAGE_HEIGHT = 356
    TARGET_IMAGE_WIDTH = 634
    INPUT_CHANNELS = 3
    
    # 實際資料集路徑
    TIMESTAMP_NAME = '20250728_184331'
    EVENT_PATH = f'C:/Users/User/source/repos/Car02/Event/Event_{TIMESTAMP_NAME}.txt'
    IMAGE_FOLDER = f'C:/Users/User/source/repos/Car02/Cam01/{TIMESTAMP_NAME}'
    
    LABEL_RANGES = {
        'angle': {'min': -30.0, 'max': 30.0},
        'throttle': {'min': 0.0, 'max': 1.0}
    }
    
    @staticmethod
    def norm_value(value, data_min, data_max, target_min=0.0, target_max=1.0):
        """正規化函數，將數值縮放到目標範圍"""
        if data_max - data_min == 0:
            return 0.0
        normalized_value = (value - data_min) / (data_max - data_min)
        return normalized_value * (target_max - target_min) + target_min

config = MockConfig()
input_shape = (config.TARGET_IMAGE_HEIGHT, config.TARGET_IMAGE_WIDTH, config.INPUT_CHANNELS)

# =========================================================================
# 建立一個只處理圖像的簡化模型
# =========================================================================
def create_simplified_cnn_model(input_shape):
    """
    創建一個簡化的 CNN 模型，只用於圖像輸入，並輸出轉向角。
    此版本增加了更多卷積層，並調整了卷積核大小，以期提高學習能力。
    """
    model = models.Sequential(name='Simplified_Model')
    
    # 標準化影像
    model.add(layers.Lambda(lambda img: img / 255.0, name='normalization', input_shape=input_shape))
    
    # 卷積層
    model.add(layers.Conv2D(32, (5, 5), strides=2, padding='valid', activation='elu', name='conv1'))
    model.add(layers.Conv2D(64, (5, 5), strides=2, padding='valid', activation='elu', name='conv2'))
    model.add(layers.Conv2D(128, (5, 5), strides=2, padding='valid', activation='elu', name='conv3'))
    model.add(layers.Conv2D(256, (3, 3), strides=2, padding='valid', activation='elu', name='conv4'))
    
    # 全連接層
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(512, activation='elu', name='fc1'))
    model.add(layers.Dropout(0.5, name='dropout1'))
    model.add(layers.Dense(256, activation='elu', name='fc2'))
    model.add(layers.Dropout(0.5, name='dropout2'))
    
    # 轉向角輸出層
    model.add(layers.Dense(1, activation='tanh', name='steering_output'))
    
    return model

# =========================================================================
# 載入真實資料集 (已根據使用者提供的格式更新)
# =========================================================================
def load_real_data(event_path, image_folder, image_shape=input_shape):
    """
    根據使用者提供的檔案格式，從指定的檔案和資料夾載入真實影像與轉向角資料。
    """
    if not os.path.exists(event_path):
        print(f"❌ 錯誤：找不到事件資料檔案：{event_path}")
        return None, None
        
    if not os.path.exists(image_folder):
        print(f"❌ 錯誤：找不到影像資料夾：{image_folder}")
        return None, None

    images = []
    steering_angles = []
    
    # 讀取事件 log 檔案
    with open(event_path, 'r') as f:
        lines = f.readlines()
    
    print(f"✅ 成功讀取 {len(lines)} 行事件資料。")
    
    for line in lines:
        line = line.strip()
        # 跳過標頭或空行
        if line.startswith('Timestamp') or not line:
            continue
            
        try:
            parts = line.split(';')
            timestamp_str = parts[0]
            event_value_str = parts[2]
            
            # 從 event_value_str 中解析出轉向角
            steering_angle_str = event_value_str.split(',')[0]
            steering_angle = float(steering_angle_str)
            
            # 根據 timestamp 構建圖像路徑
            image_filename = f"Screenshot_{timestamp_str}.jpg"
            img_path = os.path.join(image_folder, image_filename)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # 調整影像大小以符合模型輸入
                    img = cv2.resize(img, (image_shape[1], image_shape[0]))
                    images.append(img)
                    steering_angles.append(steering_angle)
                else:
                    print(f"⚠️ 警告：無法讀取影像檔案：{img_path}")
            # else:
            #     print(f"⚠️ 警告：找不到對應的影像檔案：{img_path}")
                
        except (IndexError, ValueError) as e:
            print(f"⚠️ 警告：解析行 '{line}' 時發生錯誤：{e}")
            continue

    if not images:
        print("❌ 錯誤：沒有影像被成功載入。請檢查檔案格式與內容。")
        return None, None

    images = np.array(images)
    steering_angles = np.array(steering_angles, dtype=np.float32).reshape(-1, 1)

    # 正規化轉向角
    steering_angles_normed = MockConfig.norm_value(steering_angles, 
                                                   MockConfig.LABEL_RANGES['angle']['min'], 
                                                   MockConfig.LABEL_RANGES['angle']['max'],
                                                   target_min=-1.0,
                                                   target_max=1.0)
    
    print(f"✅ 成功載入 {len(images)} 筆真實影像與轉向角資料。")
    return images, steering_angles_normed

# =========================================================================
# 視覺化訓練過程中的模型預測
# =========================================================================
def visualize_predictions(model, val_images, val_angles, epoch, total_epochs, num_samples=5):
    """
    視覺化模型的預測結果。
    """
    
    if num_samples > len(val_images):
        num_samples = len(val_images)
        
    sample_indices = random.sample(range(len(val_images)), num_samples)
    
    predictions = model.predict(val_images[sample_indices], verbose=0)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    fig.suptitle(f"Epoch {epoch}/{total_epochs} - 模型預測視覺化", fontsize=16)

    for i, idx in enumerate(sample_indices):
        ax = axes[i] if num_samples > 1 else axes
        
        real_angle_normed = val_angles[idx][0]
        predicted_angle_normed = predictions[i][0]

        real_angle = MockConfig.norm_value(real_angle_normed, -1.0, 1.0,
                                         MockConfig.LABEL_RANGES['angle']['min'],
                                         MockConfig.LABEL_RANGES['angle']['max'])
        predicted_angle = MockConfig.norm_value(predicted_angle_normed, -1.0, 1.0,
                                             MockConfig.LABEL_RANGES['angle']['min'],
                                             MockConfig.LABEL_RANGES['angle']['max'])
                                             
        ax.imshow(cv2.cvtColor(val_images[idx], cv2.COLOR_BGR2RGB))
        ax.set_title(f"真值: {real_angle:.2f}\n預測: {predicted_angle:.2f}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# =========================================================================
# 主訓練流程
# =========================================================================
if __name__ == "__main__":
    # 載入真實資料集
    images, angles = load_real_data(config.EVENT_PATH, config.IMAGE_FOLDER)
    
    # 檢查是否成功載入數據
    if images is None or images.shape[0] == 0:
        print("❌ 無法載入任何數據，訓練終止。請檢查檔案路徑和格式是否正確。")
    else:
        # 分割訓練集與驗證集
        split_index = int(len(images) * 0.9)
        train_images, val_images = images[:split_index], images[split_index:]
        train_angles, val_angles = angles[:split_index], angles[split_index:]
        
        print(f"訓練樣本數: {len(train_images)}")
        print(f"驗證樣本數: {len(val_images)}")
        
        # 建立模型
        model = create_simplified_cnn_model(input_shape)
        model.summary()
        
        # 編譯模型
        model.compile(optimizer='adam', loss='mse')
        print("✅ 模型已編譯完成！")
        
        # 開始訓練
        total_epochs = 10
        
        for epoch in range(total_epochs):
            print(f"\n--- Epoch {epoch+1}/{total_epochs} ---")
            
            history = model.fit(train_images, train_angles, 
                                epochs=1, 
                                batch_size=32, 
                                verbose=1,
                                validation_data=(val_images, val_angles))
            
            # 視覺化訓練過程
            visualize_predictions(model, val_images, val_angles, epoch + 1, total_epochs, num_samples=5)
        
        print("\n✅ 訓練完成，請觀察上面的圖表來判斷模型是否能學習。")
