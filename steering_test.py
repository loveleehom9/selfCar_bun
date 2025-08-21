# -*- coding: utf-8 -*-
"""
這個腳本是在前一個版本的基礎上，加入了數據平衡的功能，並新增了訓練過程中
損失曲線的視覺化。這將幫助我們更直觀地判斷模型是否正在學習。
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from math import ceil
from collections import Counter

# =========================================================================
# 模擬 config 模組和設定
# =========================================================================
class MockConfig:
    # 圖像目標高度與寬度
    TARGET_IMAGE_HEIGHT = 66
    TARGET_IMAGE_WIDTH = 200
    INPUT_CHANNELS = 3
    
    # 實際資料集路徑 (請根據您的實際路徑修改)
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

# 設定 GPU 記憶體使用，避免 OOM 錯誤
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# =========================================================================
# 建立一個只處理圖像的簡化模型 (沿用使用者提供的模型架構)
# =========================================================================
def create_simplified_cnn_model(input_shape):
    """
    創建一個簡化的 CNN 模型，只用於圖像輸入，並輸出轉向角。
    此版本增加了更多卷積層，並調整了卷積核大小，以期提高學習能力。
    """
    model = models.Sequential(name='Simplified_Model')
    
    model.add(layers.Conv2D(32, (5, 5), strides=2, padding='valid', activation='elu', input_shape=input_shape, name='conv1'))
    model.add(layers.Conv2D(64, (5, 5), strides=2, padding='valid', activation='elu', name='conv2'))
    model.add(layers.Conv2D(128, (5, 5), strides=2, padding='valid', activation='elu', name='conv3'))
    model.add(layers.Conv2D(256, (3, 3), strides=2, padding='valid', activation='elu', name='conv4'))
    
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(512, activation='elu', name='fc1'))
    model.add(layers.Dropout(0.5, name='dropout1'))
    model.add(layers.Dense(256, activation='elu', name='fc2'))
    model.add(layers.Dropout(0.5, name='dropout2'))
    
    model.add(layers.Dense(1, activation='tanh', name='steering_output'))
    
    return model

# =========================================================================
# 載入真實資料集路徑 (修正：加入數據平衡功能)
# =========================================================================
def load_data_paths(event_path, image_folder, balance_data=True):
    """
    根據使用者提供的檔案格式，從指定的檔案和資料夾載入真實影像與轉向角資料的路徑。
    此版本增加了數據平衡功能，通過過度採樣來處理數據不平衡問題。
    """
    if not os.path.exists(event_path):
        print(f"❌ 錯誤：找不到事件資料檔案：{event_path}")
        return None, None
        
    if not os.path.exists(image_folder):
        print(f"❌ 錯誤：找不到影像資料夾：{image_folder}")
        return None, None

    image_paths_raw = []
    steering_angles_raw = []
    
    # 讀取事件 log 檔案
    with open(event_path, 'r') as f:
        lines = f.readlines()
    
    print(f"✅ 成功讀取 {len(lines)} 行事件資料。")
    
    valid_data_count = 0
    invalid_data_count = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith('Timestamp') or not line:
            continue
            
        try:
            parts = line.split(';')
            timestamp_str = parts[0]
            event_value_str = parts[2]
            
            values = event_value_str.split(',')

            steering_angle_str = values[2].strip()
            steering_angle = float(steering_angle_str)
            
            min_angle = MockConfig.LABEL_RANGES['angle']['min']
            max_angle = MockConfig.LABEL_RANGES['angle']['max']
            if min_angle <= steering_angle <= max_angle:
                image_filename = f"Screenshot_{timestamp_str}.jpg"
                img_path = os.path.join(image_folder, image_filename)
                
                if os.path.exists(img_path):
                    image_paths_raw.append(img_path)
                    steering_angles_raw.append(steering_angle)
                    valid_data_count += 1
            else:
                invalid_data_count += 1
                
        except (IndexError, ValueError) as e:
            print(f"⚠️ 警告：解析行 '{line}' 時發生錯誤：{e}")
            invalid_data_count += 1
            continue

    if not image_paths_raw:
        print("❌ 錯誤：沒有影像路徑被成功載入。請檢查檔案格式與內容。")
        return None, None

    print(f"✅ 成功載入 {valid_data_count} 筆原始有效數據。")
    print(f"✅ 共有 {invalid_data_count} 筆無效數據被忽略。")

    if balance_data:
        print("--- 正在平衡數據集 ---")
        
        # 建立數據桶（bins）
        bins = np.linspace(min(steering_angles_raw), max(steering_angles_raw), num=50)
        
        # 將每個轉向角分配到對應的桶中
        indices = np.digitize(steering_angles_raw, bins)
        
        # 找出每個桶的數據量
        counts = Counter(indices)
        
        # 找出數據量最多的桶（通常是轉向角為 0 的桶）
        max_count = max(counts.values())
        
        # 進行過度採樣
        image_paths_balanced = []
        steering_angles_balanced = []
        
        for i, count in counts.items():
            if count < max_count:
                # 找出屬於這個桶的所有數據
                bin_indices = [j for j, idx in enumerate(indices) if idx == i]
                # 從這個桶的數據中隨機抽樣，直到數量與 max_count 相等
                if bin_indices:
                    oversample_indices = np.random.choice(bin_indices, size=max_count, replace=True)
                    for oversample_idx in oversample_indices:
                        image_paths_balanced.append(image_paths_raw[oversample_idx])
                        steering_angles_balanced.append(steering_angles_raw[oversample_idx])
            else:
                # 對於數量充足的桶，直接加入所有數據
                for j, idx in enumerate(indices):
                    if idx == i:
                        image_paths_balanced.append(image_paths_raw[j])
                        steering_angles_balanced.append(steering_angles_raw[j])
                        
        image_paths = image_paths_balanced
        steering_angles = steering_angles_balanced
        
        print(f"✅ 數據平衡完成，總樣本數為 {len(image_paths)} 筆。")
        
    else:
        image_paths = image_paths_raw
        steering_angles = steering_angles_raw

    # 正規化轉向角，並將其轉換為 NumPy 陣列
    steering_angles_normed = MockConfig.norm_value(
        np.array(steering_angles, dtype=np.float32),
        MockConfig.LABEL_RANGES['angle']['min'],
        MockConfig.LABEL_RANGES['angle']['max'],
        target_min=-1.0,
        target_max=1.0
    ).reshape(-1, 1)

    return image_paths, steering_angles_normed

# =========================================================================
# 數據生成器 (使用生成器按批次載入數據)
# =========================================================================
def data_generator(image_paths, steering_angles, batch_size, image_shape):
    """
    一個 Python 生成器，用於按批次載入、預處理和返回影像與標籤。
    """
    num_samples = len(image_paths)
    while True:
        # 打亂數據以確保每個 epoch 的順序不同
        from sklearn.utils import shuffle
        image_paths, steering_angles = shuffle(image_paths, steering_angles)
        
        for offset in range(0, num_samples, batch_size):
            batch_image_paths = image_paths[offset:offset+batch_size]
            batch_steering_angles = steering_angles[offset:offset+batch_size]
            
            batch_images = []
            for img_path in batch_image_paths:
                img = cv2.imread(img_path)
                if img is not None:
                    # 調整影像大小以符合模型輸入
                    img = cv2.resize(img, (image_shape[1], image_shape[0]))
                    # 轉換為 RGB 格式 (OpenCV 預設為 BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # 正規化：將像素值從 0-255 縮放到 0-1
                    img = img.astype(np.float32) / 255.0
                    batch_images.append(img)
            
            # 將列表轉換為 NumPy 陣列
            X = np.array(batch_images)
            y = np.array(batch_steering_angles)
            
            yield X, y

# =========================================================================
# 繪製訓練歷史曲線
# =========================================================================
def plot_history(history):
    """
    繪製訓練與驗證損失曲線，以視覺化模型的學習進度。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='訓練損失')
    plt.plot(history.history['val_loss'], label='驗證損失')
    plt.title('訓練與驗證損失', fontsize=16)
    plt.xlabel('代數 (Epoch)', fontsize=12)
    plt.ylabel('損失 (Loss)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    print("✅ 訓練歷史曲線圖已繪製。")

# =========================================================================
# 繪製轉向角數據分佈
# =========================================================================
def plot_steering_angle_distribution(steering_angles, title):
    """
    繪製轉向角的直方圖，以視覺化數據的分佈。
    """
    if steering_angles is None or len(steering_angles) == 0:
        print(f"⚠️ 警告：沒有 {title} 數據可以繪製直方圖。")
        return
        
    actual_angles = MockConfig.norm_value(
        steering_angles, -1.0, 1.0,
        MockConfig.LABEL_RANGES['angle']['min'],
        MockConfig.LABEL_RANGES['angle']['max']
    )
    
    plt.figure(figsize=(10, 6))
    plt.hist(actual_angles, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'{title} 轉向角數據分佈', fontsize=16)
    plt.xlabel('轉向角', fontsize=12)
    plt.ylabel('數據點數量', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    print(f"✅ {title} 轉向角數據分佈圖已繪製。")

# =========================================================================
# 視覺化訓練過程中的模型預測 (沿用使用者提供的函數)
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
                                             
        ax.imshow(val_images[idx]) # 影像已在生成器中正規化為 RGB，所以不需要 cvtColor
        ax.set_title(f"真值: {real_angle:.2f}\n預測: {predicted_angle:.2f}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# =========================================================================
# 主訓練流程
# =========================================================================
if __name__ == "__main__":
    
    print("--- 載入並平衡數據路徑中 ---")
    image_paths, angles = load_data_paths(config.EVENT_PATH, config.IMAGE_FOLDER, balance_data=True)
    
    if image_paths is None or len(image_paths) == 0:
        print("❌ 無法載入任何數據，訓練終止。請檢查檔案路徑和格式是否正確。")
    else:
        # 在訓練前先視覺化數據分佈
        plot_steering_angle_distribution(angles, "平衡後")

        # 分割訓練集與驗證集的路徑
        X_train_paths, X_val_paths, y_train, y_val = train_test_split(
            image_paths, angles, test_size=0.1, random_state=42
        )
        
        print(f"訓練樣本數: {len(X_train_paths)}")
        print(f"驗證樣本數: {len(X_val_paths)}")
        
        # 建立模型
        model = create_simplified_cnn_model(input_shape)
        model.summary()
        
        # 編譯模型
        model.compile(optimizer='adam', loss='mse')
        print("✅ 模型已編譯完成！")
        
        # 建立訓練和驗證數據生成器
        train_generator = data_generator(X_train_paths, y_train, 32, input_shape)
        val_generator = data_generator(X_val_paths, y_val, 32, input_shape)

        # 計算每個 epoch 的步數
        train_steps = ceil(len(X_train_paths) / 32)
        val_steps = ceil(len(X_val_paths) / 32)
        
        print(f"每個 Epoch 的訓練步數: {train_steps}")
        print(f"每個 Epoch 的驗證步數: {val_steps}")
        
        # 開始訓練
        total_epochs = 10 # 為了展示，可以先設定為較小的值
        
        history = model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=total_epochs,
            validation_data=val_generator,
            validation_steps=val_steps
        )
        
        print("\n✅ 訓練完成，請觀察上面的圖表來判斷模型是否能學習。")
        
        # 視覺化訓練歷史
        plot_history(history)
        
        # 為了視覺化，我們需要從驗證生成器中取出一個批次來預測
        try:
            val_batch_images, val_batch_angles = next(val_generator)
            visualize_predictions(model, val_batch_images, val_batch_angles, 1, total_epochs, num_samples=5)
        except StopIteration:
            print("⚠️ 警告：無法從驗證生成器中獲取樣本進行視覺化。")
            
