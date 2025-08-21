import tensorflow as tf
import numpy as np
import os
import cv2
import config
import random
# Add by bun for  balance data 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')

# 設定寬度與高度及通道資訊
input_height = config.TARGET_IMAGE_HEIGHT
input_width = config.TARGET_IMAGE_WIDTH
input_channels = config.INPUT_CHANNELS
sensor_dim = config.SENSOR_INPUT_DIM

# --- 數據增強參數設定 ---
# 隨機水平偏移的校正因子，一般是 0.002 到 0.005
# 如果圖像向左偏移 100 個像素，轉向角要增加 100 * STEER_CORRECTION_FACTOR
STEER_CORRECTION_FACTOR = 0.004 # 0.004
# 轉向角噪音的標準差
STEERING_NOISE_STD = 0.005 # 噪音不能太大 0.005

BRIGHTNESS_RANGE = (0.7, 1.3)  # 亮度調整範圍 (乘法因子)
CONTRAST_RANGE = (0.8, 1.2)    # 對比度調整範圍 (乘法因子)
HUE_RANGE = (-10, 10)          # 色度調整偏移量範圍 
SATURATION_RANGE = (0.7, 1.3)  # 飽和度調整範圍 (乘法因子)
MAX_HORIZONTAL_SHIFT_RATIO = 0.15 # 最大水平偏移佔圖像寬度的比例 ( 0.15 表示最大 15%)
SENSOR_NOISE_STD = 0.005       # 感測器數據噪音的標準差 (在正規化後的 [0,1] 或 [-1,1] 範圍內)

# --- 數據增強生成數量設定 ---
AUGMENTATIONS_PER_ORIGINAL_SAMPLE = 500 # 每個原始樣本額外生成多少個增強樣本

def parse_txt_file(txt_path):
    file_list = []
    with open(txt_path, 'r') as f:
        for line in f:
            if line.strip() == '':  # 跳過空行
                continue
            parts = line.strip().split(';')  # 使用分號分隔

            # 確保有足夠的資料
            if len(parts) >= 11:
                img_path = parts[0]
                try:
                    # 解析 angle 和 throttle
                    angle = float(parts[1])
                    throttle = float(parts[2])

                    # 解析 velocity (speed)
                    velocity = float(parts[3])

                    # 解析 rotation (x, y, z) 這是感測器的方向
                    rotation = parts[4].strip('()').split(',')
                    rotation = [float(val) for val in rotation]  # 去除括號並轉為 float

                    # 解析 angular_velocity (x, y, z) 這是角速度
                    angular_velocity = parts[5].strip('()').split(',')
                    angular_velocity = [float(val) for val in angular_velocity]  # 去除括號並轉為 float

                    # 解析 rollAngle, distFront, distRear, distLeft, distRight
                    rollAngle = float(parts[6])
                    distFront = float(parts[7])
                    distRear = float(parts[8])
                    distLeft = float(parts[9])
                    distRight = float(parts[10])

                    # 添加資料
                    file_list.append((img_path, angle, throttle, velocity, rotation, angular_velocity, rollAngle, distFront, distRear, distLeft, distRight))
                except ValueError as e:
                    print(f"❌ 解析錯誤行：{line.strip()}，錯誤：{e}")
                    print(f"角度: {parts[1]}，油門: {parts[2]}")
                    print(f"velocity: {parts[3]}, rotation: {parts[4]}, angular_velocity: {parts[5]}")
                    continue  # 跳過當前行，繼續處理下一行
            else:
                print(f"⚠️ 格式錯誤，資料欄位不足：{line.strip()}")
    return file_list


def preprocess(img):
    # 圖像預處理，可擴充為 Resize, Normalize, etc.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.resize(img, (input_width, input_height))
    # img = img / 255.0 # 先不正規化，讓數據增強時方便操作
    return img.astype(np.float32)

# 將 list資料裝載至字典內
def sensor_list_to_dict(sensor_list):
    """
    將感測器數據列表轉換為字典格式。
    """
    sensor_names = [
        "rot_x", "rot_y", "rot_z", 
        "ang_vel_x", "ang_vel_y", "ang_vel_z", 
        "speed", "rollAngle", 
        "distFront", "distRear", "distLeft", "distRight"
    ]
    
    if len(sensor_list) != len(sensor_names):
        raise ValueError(f"感測器列表長度不匹配。預期 {len(sensor_names)} 個，但得到 {len(sensor_list)} 個。")
        
    return dict(zip(sensor_names, sensor_list))

"""
# =========================================================================
# 數據增強函數 Add by bun in 08/04
# =========================================================================
def augment_image_and_data(image_hsv_raw, raw_angle, raw_throttle, raw_sensor_data):
    
    對單個樣本進行數據增強，包括圖像增強和標籤調整。

    Args:
        image_hsv_raw (np.ndarray): 原始圖像 (HSV 格式, 0-255)。
        raw_angle (float): 原始轉向角 (原始單位，未正規化)。
        raw_throttle (float): 原始油門 (原始單位，未正規化)。
        raw_sensor_data (list or tuple): 原始感測器數據 (已展開，未正規化)。
            順序為 [rot_x, rot_y, rot_z, ang_vel_x, ang_vel_y, ang_vel_z,
                speed, rollAngle, distFront, distRear, distLeft, distRight]

    Returns:
        tuple: (augmented_image, augmented_angle, augmented_throttle, augmented_sensor_data)
               所有返回數據均為原始單位，未正規化。
               augmented_image 是 RGB 格式，0-255。
    
    augmented_image_hsv = np.copy(image_hsv_raw)
    augmented_angle = float(raw_angle)
    augmented_throttle = float(raw_throttle)
    augmented_sensor_data = list(raw_sensor_data) # 複製一份感測器數據以便修改

    # 製作字典檔
    raw_data = {
            'steering': raw_angle,
            'throttle': raw_throttle,
            'sensor_data': sensor_list_to_dict(raw_sensor_data)
        }

    # 1. 隨機水平翻轉 (50% 機率)
    if random.random() < 0.5:
        augmented_image_hsv = cv2.flip(augmented_image_hsv, 1) # 水平翻轉圖像

        augmented_angle = -augmented_angle # 轉向角取負

        # 調整感測器數據以反映翻轉
        temp_sensor_data = list(raw_sensor_data) # 複製原始數據，用於翻轉的基準
        
        # rotation_xyz (索引 0, 1, 2)
        # 旋轉角度調整 (Yaw 軸) 類似取負值
        augmented_sensor_data[1] = -temp_sensor_data[1] # Yaw (Rotation Y)

        # angular_velocity_xyz (索引 3, 4, 5)
        augmented_sensor_data[4] = -temp_sensor_data[4] # Angular Velocity Y (Yaw Rate) 取負
        augmented_sensor_data[5] = -temp_sensor_data[5] # Angular Velocity Z (Roll Rate) 取負

        # 距離感測器左右互換 (索引 8, 9) (在 raw_sensor_data_combined 中，distLeft 是索引 10, distRight 是索引 11)
        # 這裡的索引是基於傳入 augmented_sensor_data 列表的索引
        # raw_sensor_data_combined 順序為:
        # [rot_x, rot_y, rot_z, ang_vel_x, ang_vel_y, ang_vel_z, speed, rollAngle, distFront, distRear, distLeft, distRight]
        # 索引 8 是 distFront, 索引 9 是 distRear, 索引 10 是 distLeft, 索引 11 是 distRight
        temp_distLeft = augmented_sensor_data[10]
        augmented_sensor_data[10] = augmented_sensor_data[11]
        augmented_sensor_data[11] = temp_distLeft
        # Roll Angle (索引 7) 通常不會因為水平翻轉而改變

        augmented_data = {
            'steering': augmented_angle,
            'throttle': augmented_throttle,
            'sensor_data': sensor_list_to_dict(augmented_sensor_data)
        }

        #compare_original_and_augmented_data(image_hsv_raw, raw_angle, raw_throttle, raw_sensor_data,\
        #    augmented_image_hsv,augmented_angle,augmented_throttle,augmented_sensor_data)
        #print(raw_data)
        #print(type(raw_data))
        
        #print(augmented_sensor_data)
        #print(type(augmented_sensor_data))
        # save_comparison_results(image_hsv_raw,augmented_image_hsv,raw_data,augmented_data)

    # 2. 隨機水平偏移 (調整圖像和轉向角)
    max_offset_pixels = int(input_width * MAX_HORIZONTAL_SHIFT_RATIO)
    offset_pixels = random.randint(-max_offset_pixels, max_offset_pixels)
    
    M = np.float32([[1, 0, offset_pixels], [0, 1, 0]])
    augmented_image_hsv = cv2.warpAffine(augmented_image_hsv, M, (input_width, input_height),
                                         borderMode=cv2.BORDER_REPLICATE) # 用邊界像素填充

    # 調整轉向角：如果圖像向左移 (offset_pixels < 0)，車輛看起來偏右，需要向左轉 (steering_angle 增加)
    # 反之，如果圖像向右移 (offset_pixels > 0)，車輛看起來偏左，需要向右轉 (steering_angle 減少)
    augmented_angle = augmented_angle - (offset_pixels * STEER_CORRECTION_FACTOR)
    augmented_angle = np.clip(augmented_angle, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'])

    save_comparison_results(image_hsv_raw,augmented_image_hsv,raw_data,augmented_data)

    # 3. 圖像亮度調整 (在 HSV 空間調整 V 通道)
    brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
    augmented_image_hsv[:,:,2] = np.clip(augmented_image_hsv[:,:,2] * brightness_factor, 0, 255)

    # 4. 圖像飽和度調整 (在 HSV 空間調整 S 通道)
    saturation_factor = random.uniform(*SATURATION_RANGE)
    augmented_image_hsv[:,:,1] = np.clip(augmented_image_hsv[:,:,1] * saturation_factor, 0, 255)

    # 5. 圖像色度調整 (在 HSV 空間調整 H 通道)
    hue_offset = random.randint(*HUE_RANGE) # 隨機偏移量，例如 -10 到 10
    augmented_image_hsv[:,:,0] = (augmented_image_hsv[:,:,0] + hue_offset) % 180 # HSV 的 H 範圍是 0-179

    # 6. 圖像對比度調整 (需要轉回 RGB 處理，或使用 skimage exposure)
    # 為了簡化，我們在 HSV 增強後統一轉回 RGB 進行對比度調整
    augmented_image_rgb = cv2.cvtColor(augmented_image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    contrast_factor = random.uniform(*CONTRAST_RANGE)
    mean = np.mean(augmented_image_rgb)
    augmented_image_rgb = np.clip((augmented_image_rgb - mean) * contrast_factor + mean, 0, 255)
    augmented_image_rgb = augmented_image_rgb.astype(np.float32) # 保持浮點數類型

    # 7. 對轉向角標籤添加少量隨機噪聲
    augmented_angle += np.random.normal(0, STEERING_NOISE_STD)
    augmented_angle = np.clip(augmented_angle, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'])

    # 8. 對感測器數據添加少量隨機噪聲 (在正規化前添加)
    # augmented_sensor_data += np.random.normal(0, SENSOR_NOISE_STD, len(augmented_sensor_data))
    # Note: 這裡暫時不對感測器數據加噪聲，因為其範圍差異大，統一的 STD 可能不合適。
    # 如果要加，應該針對每個感測器類型設定不同的噪聲標準差。

    return augmented_image_rgb, augmented_angle, augmented_throttle, augmented_sensor_data
### 新增: augment_image_and_data 函數結束
"""

# =========================================================================
# 數據增強操作
# 每個函數只執行一種增強，並返回新的圖片和數據
# =========================================================================

def _augment_flip(image_hsv, angle, throttle, sensor_data_list):
    """水平翻轉圖片、轉向角和相關感測器數據"""
    augmented_image_hsv = cv2.flip(image_hsv, 1)
    augmented_angle = -angle
    augmented_sensor_data = list(sensor_data_list) # 複製一份以便修改

    augmented_sensor_data[1] = (360 - augmented_sensor_data[1]) % 360 # rot_y
    augmented_sensor_data[4] = -augmented_sensor_data[4] # ang_vel_y
    augmented_sensor_data[5] = -augmented_sensor_data[5] # ang_vel_z
    
    temp_distLeft = augmented_sensor_data[10]
    augmented_sensor_data[10] = augmented_sensor_data[11] # distLeft
    augmented_sensor_data[11] = temp_distLeft # distRight

    return augmented_image_hsv, augmented_angle, throttle, augmented_sensor_data

def _augment_shift(image_hsv, angle, throttle, sensor_data_list):
    """隨機水平偏移圖片並調整轉向角"""
    augmented_image_hsv = np.copy(image_hsv)
    augmented_angle = angle
    augmented_throttle = throttle
    augmented_sensor_data = list(sensor_data_list)

    max_offset_pixels = int(input_width * MAX_HORIZONTAL_SHIFT_RATIO)
    offset_pixels = random.randint(-max_offset_pixels, max_offset_pixels)
    
    M = np.float32([[1, 0, offset_pixels], [0, 1, 0]])
    augmented_image_hsv = cv2.warpAffine(augmented_image_hsv, M, (input_width, input_height),
                                         borderMode=cv2.BORDER_REPLICATE)
    
    augmented_angle = augmented_angle - (offset_pixels * STEER_CORRECTION_FACTOR)
    augmented_angle = np.clip(augmented_angle, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'])

    return augmented_image_hsv, augmented_angle, augmented_throttle, augmented_sensor_data

def _augment_brightness(image_hsv, angle, throttle, sensor_data_list):
    """隨機調整圖片亮度 (HSV V通道)"""
    augmented_image_hsv = np.copy(image_hsv)
    brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
    augmented_image_hsv[:,:,2] = np.clip(augmented_image_hsv[:,:,2] * brightness_factor, 0, 255)
    
    return augmented_image_hsv, angle, throttle, sensor_data_list

def _augment_saturation(image_hsv, angle, throttle, sensor_data_list):
    """隨機調整圖片飽和度 (HSV S通道)"""
    augmented_image_hsv = np.copy(image_hsv)
    saturation_factor = random.uniform(*SATURATION_RANGE)
    augmented_image_hsv[:,:,1] = np.clip(augmented_image_hsv[:,:,1] * saturation_factor, 0, 255)
    
    return augmented_image_hsv, angle, throttle, sensor_data_list

def _augment_hue(image_hsv, angle, throttle, sensor_data_list):
    """隨機調整圖片色度 (HSV H通道)"""
    augmented_image_hsv = np.copy(image_hsv)
    hue_offset = random.randint(*HUE_RANGE)
    augmented_image_hsv[:,:,0] = (augmented_image_hsv[:,:,0] + hue_offset) % 180
    
    return augmented_image_hsv, angle, throttle, sensor_data_list

def _augment_contrast(image_hsv, angle, throttle, sensor_data_list):
    """隨機調整圖片對比度 (RGB空間處理後轉回HSV)"""
    augmented_image_hsv = np.copy(image_hsv)
    
    # 轉換到 RGB 進行對比度調整
    temp_rgb = cv2.cvtColor(augmented_image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    contrast_factor = random.uniform(*CONTRAST_RANGE)
    mean = np.mean(temp_rgb)
    temp_rgb = np.clip((temp_rgb - mean) * contrast_factor + mean, 0, 255)
    
    # 轉換回 HSV
    augmented_image_hsv = cv2.cvtColor(temp_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    
    return augmented_image_hsv, angle, throttle, sensor_data_list

def _add_steering_noise(image_hsv, angle, throttle, sensor_data_list):
    """對轉向角添加少量隨機噪聲"""
    augmented_image_hsv = np.copy(image_hsv) # 圖像不變
    augmented_angle = angle + np.random.normal(0, STEERING_NOISE_STD)
    augmented_angle = np.clip(augmented_angle, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'])
    
    return augmented_image_hsv, augmented_angle, throttle, sensor_data_list

# 所有可用的增強函數列表
AUGMENTATION_FUNCTIONS = [
    _augment_flip,
    _augment_shift,
    _augment_brightness,
    _augment_saturation,
    _augment_hue,
    _augment_contrast,
    _add_steering_noise
]

def apply_one_random_augmentation(image_hsv_raw, raw_angle, raw_throttle, raw_sensor_data_combined):
    """
    隨機選擇並應用一種增強。
    
    Args:
        image_hsv_raw (np.ndarray): 原始圖像 (HSV 格式, 0-255)。
        raw_angle (float): 原始轉向角。
        raw_throttle (float): 原始油門。
        raw_sensor_data_combined (list): 原始感測器數據列表。
        
    Returns:
        tuple: (augmented_image_bgr, augmented_angle, augmented_throttle, augmented_sensor_data_list)
               所有返回數據均為原始單位。augmented_image_bgr 是 BGR 格式，0-255。
    """
    # 隨機選擇一種增強函數
    chosen_augmentation_func = random.choice(AUGMENTATION_FUNCTIONS)
    
    # 應用選定的增強
    augmented_image_hsv, augmented_angle, augmented_throttle, augmented_sensor_data_list = \
        chosen_augmentation_func(image_hsv_raw, raw_angle, raw_throttle, raw_sensor_data_combined)
    
    # 將最終的圖片從 HSV 轉回 BGR (因為 cv2.imread 讀取的是 BGR 格式，保持一致性)
    augmented_image_bgr = cv2.cvtColor(augmented_image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return augmented_image_bgr, augmented_angle, augmented_throttle, augmented_sensor_data_list


# -------------------------------------------------------------------------
# 儲存增強數據的比較資訊
# -------------------------------------------------------------------------
def save_comparison_results(raw_image_rgb, augmented_image_rgb, raw_data, augmented_data, output_folder="comparison_results"):
    """
    將原始圖片和增強後的圖片並排繪製，並將結果儲存為 PNG 檔案。
    同時將數據差異儲存為文字檔案。

    Args:
        raw_image_rgb (np.ndarray): 原始圖片 (RGB 格式, 0-255)。
        augmented_image_rgb (np.ndarray): 增強後的圖片 (RGB 格式, 0-255)。
        raw_data (dict): 包含原始轉向、油門和感測器數據的字典。
        augmented_data (dict): 包含增強後轉向、油門和感測器數據的字典。
        output_folder (str): 儲存結果的資料夾名稱。
    """
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raw_image_rgb = cv2.cvtColor(raw_image_rgb.astype(np.uint8), cv2.COLOR_HSV2RGB)
    augmented_image_rgb = cv2.cvtColor(augmented_image_rgb.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # ------------------
    # 儲存圖片比較結果
    # ------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 原始圖片 (確保是 RGB 格式，0-255)
    axes[0].imshow(raw_image_rgb.astype(np.uint8))
    axes[0].set_title(f"Org Img\nSteering: {raw_data['steering']:.4f}, Throttle: {raw_data['throttle']:.4f}")
    axes[0].axis('off')

    # 增強後的圖片 (確保是 RGB 格式，0-255)
    axes[1].imshow(augmented_image_rgb.astype(np.uint8))
    axes[1].set_title(f"Aug Img\nSteering: {augmented_data['steering']:.4f}, Throttle: {augmented_data['throttle']:.4f}")
    axes[1].axis('off')

    plt.suptitle(f"Img Compare", fontsize=16)

    # 產生一個唯一的檔名，例如使用時間戳
    timestamp = cv2.getTickCount()
    image_filename = os.path.join(output_folder, f"comparison_{timestamp}.png")
    
    plt.savefig(image_filename)
    plt.close(fig)  # 關閉圖形以釋放內存
    
    print(f"圖片比較結果已儲存至：{image_filename}")

    # ------------------
    # 儲存文字數據差異
    # ------------------
    txt_filename = os.path.join(output_folder, f"comparison_{timestamp}.txt")
    with open(txt_filename, 'w') as f:
        f.write("--- 原始數據 ---\n")
        f.write(f"轉向角: {raw_data['steering']:.4f}\n")
        f.write(f"油門: {raw_data['throttle']:.4f}\n")
        f.write("感測器數據:\n")
        for k, v in raw_data['sensor_data'].items():  # 使用巢狀字典
            f.write(f"  {k:<15}: {v:>8.4f}\n")
        f.write("\n")

        f.write("--- 增強後數據 ---\n")
        f.write(f"轉向角: {augmented_data['steering']:.4f}\n")
        f.write(f"油門: {augmented_data['throttle']:.4f}\n")
        f.write("感測器數據:\n")
        for k, v in augmented_data['sensor_data'].items():  # 使用巢狀字典
            f.write(f"  {k:<15}: {v:>8.4f}\n")
        f.write("\n")
        
        f.write("--- 數據差異 ---\n")
        """
        for k, v_orig in raw_data['sensor_data'].items(): # 使用巢狀字典
            v_aug = augmented_data['sensor_data'].get(k)
            if not np.isclose(v_orig, v_aug):
                f.write(f"  {k:<15}: 原始={v_orig:>8.4f}, 增強={v_aug:>8.4f}\n")
        """
        # 修改: 確保遍歷所有可能的鍵，即使某些鍵只存在於其中一個字典中
        all_keys = set(raw_data['sensor_data'].keys()) | set(augmented_data['sensor_data'].keys())
        for k in sorted(list(all_keys)):
            v_orig = raw_data['sensor_data'].get(k, float('nan')) # 不存在的鍵設為 NaN
            v_aug = augmented_data['sensor_data'].get(k, float('nan')) # 不存在的鍵設為 NaN
            if not np.isclose(v_orig, v_aug, equal_nan=True): # 處理 NaN 比較
                f.write(f"  {k:<15}: 原始={v_orig:>8.4f}, 增強={v_aug:>8.4f}\n")
    
    print(f"數據差異已儲存至：{txt_filename}")


# =========================================================================
# Add by bun for balance data 
# 觀測數據增強的前後差異
# =========================================================================
def compare_original_and_augmented_data(image_hsv_raw, raw_angle, raw_throttle, raw_sensor_data,\
            augmented_image_hsv,augmented_angle,augmented_throttle,augmented_sensor_data):
    """
    # --- 處理原始圖片和數據 ---
    # 預處理（轉換到 HSV 並縮放）
    #img_hsv_preprocessed = preprocess_image(img_bgr_original)
    
    # 原始圖片用於顯示
    # img_rgb_original_display = cv2.cvtColor(img_bgr_original, cv2.COLOR_BGR2RGB)
    
    # --- 顯示圖片和數據 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始圖片
    axes[0].imshow(image_hsv_raw)
    axes[0].set_title(f"原始圖片\n轉向: {raw_angle:.4f}, 油門: {raw_throttle:.4f}")
    axes[0].axis('off')
    
    # 增強後的圖片
    axes[1].imshow(augmented_image_hsv.astype(np.uint8))
    axes[1].set_title(f"增強後圖片\n轉向: {augmented_angle:.4f}, 油門: {augmented_throttle:.4f}")
    axes[1].axis('off')

    plt.suptitle(f"圖片和數據比較:\n", fontsize=16)
    plt.show()
    """
    print("\n--- 原始感測器數據 ---")
    print(f"  {raw_sensor_data}")
    print("\n--- 增強後感測器數據 ---")
    print(f"  {augmented_sensor_data}")
    print("-----------------------------------")
    print("感測器數據名稱: [rot_x, rot_y, rot_z, ang_vel_x, ang_vel_y, ang_vel_z, speed, rollAngle, distFront, distRear, distLeft, distRight]")
    print("\n--- 感測器數據差異 ---")
    sensor_names = ["rot_x", "rot_y", "rot_z", "ang_vel_x", "ang_vel_y", "ang_vel_z", 
                    "speed", "rollAngle", "distFront", "distRear", "distLeft", "distRight"]
    for name, orig, aug in zip(sensor_names, raw_sensor_data, augmented_sensor_data):
        if not np.isclose(orig, aug):
            print(f"  {name:<15}: 原始={orig:>8.4f}, 增強={aug:>8.4f}")


# =========================================================================
# Add by bun for balance data 
# 將過大的數據資料做降等處理
# =========================================================================
def balance_data(file_list, index_sensor , num_bins, extreme_values_to_balance):
    """
    從指定的數據列表中讀取數據，並根據指定的感測器數據分佈來平衡數據。
    目前採「保留相對分佈」的平衡策略：
    - 找出所有非極限值箱體中，樣本數最多的那個箱體。
    - 將所有被標記為「極限值」的箱體，其樣本數下採樣到與該最高峰相同的數量。
    - 其餘所有非極限值的箱體則保持不變。

    Args:
        file_list : 原始資料包含圖檔名稱及感測器數據。
        index_sensor : 感測器資料的index
        num_bins (int): 直方圖的區間數量。
        extreme_values_to_balance (list): 包含浮點數的列表，表示要進行下採樣的極限值。
        例如：[0.0, 1.0]
    Returns:
        list: 平衡後的數據資料。
    """
    print("正在讀取原始數據...")
    all_lines = file_list
    
    # 解析數據以獲取數據較多的資訊
    need_balance_list = []
    for line in all_lines:
        try:
            balance_data = float(line[index_sensor])
            need_balance_list.append(balance_data)
        except (ValueError, IndexError):
            continue  # 跳過格式不正確的行
    
    df = pd.DataFrame(need_balance_list, columns=['balance_data'])
    
    print(f"正在分析原始分佈 (index {index_sensor})...")
    
    # 繪製原始數據分佈圖
    plt.figure(figsize=(12, 6))
    plt.hist(df['balance_data'], bins=num_bins, color='blue', alpha=0.7)
    plt.title(f'原始分佈 (Sensor Index: {index_sensor})')
    plt.xlabel('資料')
    plt.ylabel('樣本數')
    #plt.show()

    print(f"原始樣本總數：{len(df)}")
    
    # 建立該數據的直方圖
    hist, bins = np.histogram(df['balance_data'], num_bins)
    
    # 尋找非極限值箱體的最高峰值
    max_non_extreme_count = 0
    extreme_bin_list = []

    print(f"極限值做下採樣：{extreme_values_to_balance}")

    for i in range(num_bins):
        # 直接檢查箱體的區間 [bins[i], bins[i+1]] 是否包含極限值
        is_extreme = False
        
        # 遍歷所有極限值
        for extreme_val in extreme_values_to_balance:
            # 檢查極限值是否在當前箱體的範圍內
            # 考慮浮點數誤差，使用一個小的容忍度
            bin_start = bins[i]
            bin_end = bins[i+1]
            epsilon = 1e-6 # 一個很小的數值來處理浮點數誤差

            if (extreme_val >= bin_start - epsilon and extreme_val <= bin_end + epsilon):
                is_extreme = True
                extreme_bin_list.append(i)
                break
        
        if not is_extreme:
            # 找到非極限值箱體中的最高峰
            if hist[i] > max_non_extreme_count:
                max_non_extreme_count = hist[i]
                
    # 將下採樣的目標設定為非極限箱體的最高峰值
    target_count = max_non_extreme_count
    print(f"剔除極限值後，剩餘數據中的最高峰樣本數為：{target_count}")

    remove_list = []
    for j in range(num_bins):
        # 找出每個區間內的數據索引
        # 注意：np.logical_and 包含下限，但不包含上限，除了最後一個箱體會包含上限
        # 這符合 np.histogram 的預設行為
        if j == num_bins - 1:
            list_index = np.where(np.logical_and(df['balance_data'] >= bins[j], df['balance_data'] <= bins[j+1]))[0]
        else:
            list_index = np.where(np.logical_and(df['balance_data'] >= bins[j], df['balance_data'] < bins[j+1]))[0]

        # 只有被標記為極限值的箱體才進行下採樣
        if j in extreme_bin_list:
            if len(list_index) > target_count:
                remove_count = len(list_index) - target_count
                if remove_count > 0:
                    remove_index = random.sample(list(list_index), remove_count)
                    remove_list.extend(remove_index)
        # 非極限值的箱體，暫時不做任何處理，完全保留。
            
    print(f"預計移除的樣本數：{len(remove_list)}")
    
    # 建立一個索引列表，用於從原始數據中移除選定的行
    keep_list = [i for i in range(len(all_lines)) if i not in remove_list]
    balanced_lines = [all_lines[i] for i in keep_list]
    
    # random.shuffle(balanced_lines)
    print(f"平衡後樣本總數 (index {index_sensor})：{len(balanced_lines)}")

    # 繪製平衡後數據分佈圖
    balanced_angles = [float(line[index_sensor]) for line in balanced_lines]
    plt.figure(figsize=(12, 6))
    plt.hist(balanced_angles, bins=num_bins, color='green', alpha=0.7)
    plt.title(f'平衡後分佈 (Sensor Index: {index_sensor})') 
    plt.xlabel('資料')
    plt.ylabel('樣本數')
    #plt.show()
    
    return balanced_lines

# =========================================================================
# Add by bun for data balance 
# 將資料做平衡處理，去除極限值的影響
# =========================================================================
def dalance_preProcess(file_list):

    # 平衡資料步驟1: 資料預處理 - 解析並擴展多維度數據
    # 將 rotation (x,y,z) 和 angular_velocity (x,y,z) 從tuple展開成單個元素
    expanded_file_list = []
    for line in file_list:
        try:
            # line 格式: (img_path, angle, throttle, velocity, rotation_tuple, angular_velocity_tuple, rollAngle, distFront, distRear, distLeft, distRight)
            # 展開後的順序 (為 balance_data 函數提供正確的索引):
            # 0: img_path
            # 1: angle
            # 2: throttle
            # 3: velocity
            # 4: rotation_x
            # 5: rotation_y
            # 6: rotation_z
            # 7: angular_velocity_x
            # 8: angular_velocity_y
            # 9: angular_velocity_z
            # 10: rollAngle
            # 11: distFront
            # 12: distRear
            # 13: distLeft
            # 14: distRight
            # rotation = line[4] has x y z
            # angular_velocity = line[5] has x y z
            expanded_line = line[:4] + tuple(line[4]) + tuple(line[5]) + line[6:]
            expanded_file_list.append(expanded_line)
        except (ValueError, IndexError, AttributeError):
            print(f"警告：處理行時發生錯誤，跳過。行內容: {line}")
            continue

    file_list = expanded_file_list

    # 平衡資料步驟2: 資料平衡處理 - 針對每個展開後的欄位呼叫 balance_data
    # 擴展後的索引:
    # 轉向角=1, 油門=2, 速度=3, Rotation(X,Y,Z)=4~6, Angular Velocity(X,Y,Z)=7~9,
    # Roll Angle=10, distFront=11, distRear=12, distLeft=13, distRight=14
    #print("--- 開始平衡單維度資料 ---")
    # 索引 1: 轉向角
    file_list = balance_data(file_list, index_sensor=1, num_bins=60, extreme_values_to_balance=[0.0])
    # 索引 2: 油門
    file_list = balance_data(file_list, index_sensor=2, num_bins=100, extreme_values_to_balance=[0.0, 1.0])
    # 索引 3: 速度
    file_list = balance_data(file_list, index_sensor=3, num_bins=100, extreme_values_to_balance=[0.0])

    #print("\n--- 開始平衡 Rotation XYZ ---")
    # 索引 4, 5, 6: Rotation X, Y, Z
    file_list = balance_data(file_list, index_sensor=4, num_bins=360, extreme_values_to_balance=[0.0, 360.0])
    file_list = balance_data(file_list, index_sensor=5, num_bins=360, extreme_values_to_balance=[0.0, 360.0])
    file_list = balance_data(file_list, index_sensor=6, num_bins=360, extreme_values_to_balance=[0.0, 360.0])

    #print("\n--- 開始平衡 Angular Velocity XYZ ---")
    # 索引 7, 8, 9: Angular Velocity X, Y, Z
    file_list = balance_data(file_list, index_sensor=7, num_bins=20, extreme_values_to_balance=[0.0])
    file_list = balance_data(file_list, index_sensor=8, num_bins=20, extreme_values_to_balance=[0.0])
    file_list = balance_data(file_list, index_sensor=9, num_bins=20, extreme_values_to_balance=[0.0])

    #print("\n--- 開始平衡其他數據 ---")
    # 索引 10, 11, 12, 13, 14: rollAngle, dist...
    file_list = balance_data(file_list, index_sensor=10, num_bins=360, extreme_values_to_balance=[0.0]) # rollAngle平衡數據
    file_list = balance_data(file_list, index_sensor=11, num_bins=30, extreme_values_to_balance=[30.0]) # distFront平衡數據
    file_list = balance_data(file_list, index_sensor=12, num_bins=30, extreme_values_to_balance=[30.0]) # distRear平衡數據
    file_list = balance_data(file_list, index_sensor=13, num_bins=30, extreme_values_to_balance=[30.0]) # distLeft平衡數據
    file_list = balance_data(file_list, index_sensor=14, num_bins=30, extreme_values_to_balance=[30.0]) # distRight平衡數據

    # 平衡資料步驟3: 資料後處理 - 重新組裝多維度數據
    reassembled_file_list = []
    # 重新計算原始索引
    # 轉向角=1, 油門=2, 速度=3, rotation=4, angular_velocity=5, rollAngle=6, dist...=7-10
    for expanded_line in file_list:
        # 重新組裝 rotation (x, y, z)
        rotation_tuple = [expanded_line[4], expanded_line[5], expanded_line[6]]
    
        # 重新組裝 angular_velocity (x, y, z)
        angular_velocity_tuple = [expanded_line[7], expanded_line[8], expanded_line[9]]
    
        # 加入後續的單維度數據
        reassembled_line = expanded_line[:4] + (rotation_tuple,) + (angular_velocity_tuple,) + expanded_line[10:]
    
        reassembled_file_list.append(reassembled_line)

    print("\n--- 完成所有平衡和重新組裝 ---")
    print(f"最終樣本總數：{len(reassembled_file_list)}")
    print("最終資料格式範例:")
    if reassembled_file_list:
        print(reassembled_file_list[0])

    return reassembled_file_list

# =========================================================================
# Add by bun 
# 降低直走的數據
# =========================================================================
def straight_Downsampling(file_list,straight_threshold=2.0 , retention_rate = 100.0):
    """
    將數據分為直線行駛和轉彎兩類，並做數據平衡 (Downsampling)
    
    Args:
        file_list (list): 原始數據。
        straight_threshold (float): 判斷轉向角取絕對值小於此閥值算是直行。
        retention_rate (float): 保留直行的數量與轉彎數量比例。
    Returns:
        file_list (list) : 直線行駛下採樣後的數據。
    """
    # 將數據分為直線行駛和轉彎兩類，並做數據平衡 (Downsampling)
    straight_data = []
    turning_data = []

    # straight_threshold = 5.0

    for data in file_list:
        img_path, angle, *_ = data
        # 如果 Steering Angle 的絕對值小於閾值，則視為直線行駛
        if abs(angle) <= straight_threshold:
            straight_data.append(data)
        else:
            turning_data.append(data)

    print(f"📊 數據分類(轉向角)：")
    print(f" - 直線樣本數 (abs(angle) < {straight_threshold})：{len(straight_data)}")
    print(f" - 轉彎樣本數 (abs(angle) >= {straight_threshold})：{len(turning_data)}")
    # try only use 直線樣本
    
    # 3. 對直線樣本進行下採樣 (Downsampling)
    # 讓直線樣本的數量與轉彎樣本的數量的一半
    num_turning_samples = int(len(turning_data) * retention_rate/100)
    if len(straight_data) > num_turning_samples:
        # 隨機從直線樣本中抽取與轉彎樣本相同數量的數據
        random.shuffle(straight_data) # 先打亂
        downsampled_straight_data = straight_data[:num_turning_samples]
        print(f"對直線樣本進行下採樣，保留 {len(downsampled_straight_data)} 個樣本。")
    else:
        # 如果直線樣本不多於轉彎樣本，則全部保留
        downsampled_straight_data = straight_data
        print("直線樣本數不多於轉彎樣本，全部保留。")
    

    # 4. 合併並打亂最終的訓練集
    # balanced_file_list = straight_data
    balanced_file_list = turning_data + downsampled_straight_data
    random.shuffle(balanced_file_list)
    print(f"最終訓練集樣本數 (經過直線樣本下採樣)：{len(balanced_file_list)}")

    return balanced_file_list

# =========================================================================
# Add by bun for train more image 
# 增加原始 及 水平翻轉樣本增強
# 透過新增 augment=True 變數控制。
# =========================================================================
debug_counter = 0 # 測試樣本數據顯示
def load_data(txt_path , augment=True):
    # 當 augment=True 時，會做樣本增強的訓練

    # Add by bun to test balance straight and turning data on 08/01
    """
    從 TXT 檔案載入的清單資料做平衡篩選和增強，並轉換為 TensorFlow Dataset。
    
    Args:
        txt_path (str): TXT 檔案的路徑。
        augment (bool): 是否進行水平翻轉的數據增強。
    Returns:
        tf.data.Dataset: 包含平衡和增強後數據的 TensorFlow Dataset。
    """
    # 設定隨機種子為66，若是有需要重現，就可以抓到資料
    random_seed = 66
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 1. 解析完整的數據集
    file_list = parse_txt_file(txt_path)
    print(f"✅ 載入原始數據集，總樣本數：{len(file_list)}")

    if augment:
        # 執行資料平衡，去除極限值的影響
        balance_file_list = dalance_preProcess(file_list)
        # 執行數據篩選，將直走與轉彎的數據做平衡處理
        balance_file_list = straight_Downsampling(balance_file_list,2.0,50.0)

    global debug_counter # 修改全域變數
    debug_counter = 0
    if augment:
        file_list = balance_file_list
    print(f"✅ 確認複製數據集，總樣本數：{len(file_list)}")

    def _generator():
        global debug_counter # 允許修改外部函數的變數

        for data in file_list: #平衡過的數據
            # 讀取原始數據 Add by bun for org data
            img_path, raw_angle_original, raw_throttle_original, raw_speed_original, \
            raw_rotation_tuple_original, raw_angular_velocity_tuple_original, \
            raw_rollAngle_original, raw_distFront_original, raw_distRear_original, \
            raw_distLeft_original, raw_distRight_original = data
            
            # 將感測器數據組合成一個列表以便傳遞給增強函數
            raw_sensor_data_combined = [ # 新增
                raw_rotation_tuple_original[0], raw_rotation_tuple_original[1], raw_rotation_tuple_original[2],
                raw_angular_velocity_tuple_original[0], raw_angular_velocity_tuple_original[1], raw_angular_velocity_tuple_original[2],
                raw_speed_original, raw_rollAngle_original, raw_distFront_original, raw_distRear_original,
                raw_distLeft_original, raw_distRight_original
            ] # 新增

            # img_path = data[0]

            # 讀取並處理影像
            # Add by bun to test image
            img_bgr = cv2.imread(img_path) # 讀取原始圖片 (BGR 格式)
            if img_bgr is None:
                if debug_counter < 10: # 限制警告的列印次數
                    print(f"警告: 無法載入圖片 {img_path}，跳過此樣本。")
                debug_counter += 1
                continue

            # 預處理圖像，轉換為 HSV 並 resize，且尚未正規化
            img_hsv_preprocessed = preprocess(img_bgr) # 這邊的圖像數據還沒有正規化!!!
            # img_hsv_preprocessed = preprocess_image(img_bgr) # 修改: 調用新的 preprocess_image 函數

            # --- 處理原始數據 ---
            # 將 HSV 圖像轉回 RGB 進行正規化
            img_rgb_normalized = cv2.cvtColor(img_hsv_preprocessed.astype(np.uint8), cv2.COLOR_HSV2RGB) / 255.0

            # 正規化原始感測器數據 (原有的邏輯，但從 now_data_org 改為 raw_data_original)
            sensor_data_original_normed = np.array([ # 修改
                config.norm_value(raw_rotation_tuple_original[0], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max']),
                config.norm_value(raw_rotation_tuple_original[1], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max']),
                config.norm_value(raw_rotation_tuple_original[2], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max']),
                config.norm_value(raw_angular_velocity_tuple_original[0], config.SENSOR_RANGES['angular_velocity_x']['min'], config.SENSOR_RANGES['angular_velocity_x']['max'], target_min=-1.0, target_max=1.0),
                config.norm_value(raw_angular_velocity_tuple_original[1], config.SENSOR_RANGES['angular_velocity_y']['min'], config.SENSOR_RANGES['angular_velocity_y']['max'], target_min=-1.0, target_max=1.0),
                config.norm_value(raw_angular_velocity_tuple_original[2], config.SENSOR_RANGES['angular_velocity_z']['min'], config.SENSOR_RANGES['angular_velocity_z']['max'], target_min=-1.0, target_max=1.0),
                config.norm_value(raw_speed_original, config.SENSOR_RANGES['speed']['min'], config.SENSOR_RANGES['speed']['max']),
                config.norm_value(raw_rollAngle_original, config.SENSOR_RANGES['rollAngle']['min'], config.SENSOR_RANGES['rollAngle']['max'], target_min=-1.0, target_max=1.0),
                config.norm_value(raw_distFront_original, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                config.norm_value(raw_distRear_original, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                config.norm_value(raw_distLeft_original, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                config.norm_value(raw_distRight_original, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
            ], dtype=np.float32) # 修改

            # Modify by bun to change output is 2 data
            # 整合回傳數據（角度、油門）
            """
            output = np.array([
                config.norm_value(raw_angle_original, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'], target_min=-1.0, target_max=1.0),
                config.norm_value(raw_throttle_original, config.LABEL_RANGES['throttle']['min'], config.LABEL_RANGES['throttle']['max'], target_min=0.0, target_max=1.0)
            ], dtype=np.float32)
            """

            # 將輸出資料拆為兩個獨立的變數
            # steering_output = config.norm_value(raw_angle_original, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'], target_min=-1.0, target_max=1.0)
            # throttle_output = config.norm_value(raw_throttle_original, config.LABEL_RANGES['throttle']['min'], config.LABEL_RANGES['throttle']['max'], target_min=0.0, target_max=1.0)

            steering_output_original_normed = config.norm_value(raw_angle_original, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'], target_min=-1.0, target_max=1.0)
            throttle_output_original_normed = config.norm_value(raw_throttle_original, config.LABEL_RANGES['throttle']['min'], config.LABEL_RANGES['throttle']['max'], target_min=0.0, target_max=1.0)

            # --- Yield 原始數據 ---
            if sensor_data_original_normed.shape[0] == config.SENSOR_INPUT_DIM: # 確保數據維度正確才 yield
                # 返回字典形式的資料
                # 將輸出資料組合成字典，並使用 model.compile 中定義的名稱
                # yield {'image_input': img, 'sensor_input': sensor_data}, output
                yield {'image_input': img_rgb_normalized, 'sensor_input': sensor_data_original_normed}, {'steering_output': np.array([steering_output_original_normed], dtype=np.float32), 'throttle_output': np.array([throttle_output_original_normed], dtype=np.float32)}
                
                if debug_counter < 10: # 限制列印次數
                    print(f"\n--- 預覽樣本 {debug_counter + 1} (原始數據) ---")
                    print(f"圖片路徑: {img_path}")
                    print(f"預處理後圖片形狀 (原始) : {img_rgb_normalized.shape}")
                    print(f"**正規化感測器數據 ({config.SENSOR_INPUT_DIM}維): {sensor_data_original_normed}**")
                    print(f"**正規化控制輸出 (轉向角): {steering_output_original_normed}**")
                    print(f"**正規化控制輸出 (油門): {throttle_output_original_normed}**")
                debug_counter += 1
            else:
                print(f"錯誤: 樣本 {img_path} 的原始感測器數據維度不符合。預期 {config.SENSOR_INPUT_DIM}, 實際 {sensor_data.shape[0]}。跳過此樣本。")
                debug_counter += 1
                continue # 跳過不符合維度的樣本

            # --- 處理數據增強 (如果 augment 為 True) ---
            if augment:
                if ( AUGMENTATIONS_PER_ORIGINAL_SAMPLE == 0 ) :
                    print("Break to Augmentation")
                    break
                for _ in range(AUGMENTATIONS_PER_ORIGINAL_SAMPLE): # 為每個原始樣本生成多個增強樣本
                    augmented_image_bgr_raw, augmented_angle_raw, augmented_throttle_raw, augmented_sensor_data_raw = \
                        apply_one_random_augmentation(img_hsv_preprocessed, raw_angle_original, raw_throttle_original, raw_sensor_data_combined)

                # augmented_image_rgb_raw, augmented_angle_raw, augmented_throttle_raw, augmented_sensor_data_raw = \
                #     augment_image_and_data(img_hsv_preprocessed, raw_angle_original, raw_throttle_original, raw_sensor_data_combined) # 新增: 調用增強函數

                # 對增強後的圖像進行最終歸一化 (0-1)
                #augmented_image_rgb_normalized = augmented_image_rgb_raw / 255.0 # 新增: 最終歸一化
                # 對增強後的圖像進行最終歸一化 (0-1)
                augmented_image_rgb_normalized = cv2.cvtColor(augmented_image_bgr_raw, cv2.COLOR_BGR2RGB) / 255.0


                # 正規化增強後的感測器數據
                sensor_data_augmented_normed = np.array([ # 新增
                    config.norm_value(augmented_sensor_data_raw[0], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], ),
                    config.norm_value(augmented_sensor_data_raw[1], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], ),
                    config.norm_value(augmented_sensor_data_raw[2], config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], ),
                    config.norm_value(augmented_sensor_data_raw[3], config.SENSOR_RANGES['angular_velocity_x']['min'], config.SENSOR_RANGES['angular_velocity_x']['max'], target_min=-1.0, target_max=1.0),
                    config.norm_value(augmented_sensor_data_raw[4], config.SENSOR_RANGES['angular_velocity_y']['min'], config.SENSOR_RANGES['angular_velocity_y']['max'], target_min=-1.0, target_max=1.0),
                    config.norm_value(augmented_sensor_data_raw[5], config.SENSOR_RANGES['angular_velocity_z']['min'], config.SENSOR_RANGES['angular_velocity_z']['max'], target_min=-1.0, target_max=1.0),
                    config.norm_value(augmented_sensor_data_raw[6], config.SENSOR_RANGES['speed']['min'], config.SENSOR_RANGES['speed']['max']),
                    config.norm_value(augmented_sensor_data_raw[7], config.SENSOR_RANGES['rollAngle']['min'], config.SENSOR_RANGES['rollAngle']['max'], target_min=-1.0, target_max=1.0),
                    config.norm_value(augmented_sensor_data_raw[8], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                    config.norm_value(augmented_sensor_data_raw[9], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                    config.norm_value(augmented_sensor_data_raw[10], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max']),
                    config.norm_value(augmented_sensor_data_raw[11], config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
                ], dtype=np.float32) # 新增

                steering_output_augmented_normed = config.norm_value(augmented_angle_raw, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'], target_min=-1.0, target_max=1.0) # 新增
                throttle_output_augmented_normed = config.norm_value(augmented_throttle_raw, config.LABEL_RANGES['throttle']['min'], config.LABEL_RANGES['throttle']['max'], target_min=0.0, target_max=1.0) # 新增

                # 返回字典形式的資料
                # 將輸出資料組合成字典，並使用 model.compile 中定義的名稱
                # --- Yield 翻轉後的數據 ---
                if sensor_data_augmented_normed.shape[0] == config.SENSOR_INPUT_DIM: # 新增
                    # yield {'image_input': img_flipped, 'sensor_input': sensor_data_flipped}, output_flipped
                    # yield {'image_input': img_flipped, 'sensor_input': sensor_data_flipped}, {'steering_output': np.array([steering_output_flipped], dtype=np.float32), 'throttle_output': np.array([throttle_output_flipped], dtype=np.float32)}
                    
                    yield {'image_input': augmented_image_rgb_normalized, 'sensor_input': sensor_data_augmented_normed}, \
                          {'steering_output': np.array([steering_output_augmented_normed], dtype=np.float32), \
                           'throttle_output': np.array([throttle_output_augmented_normed], dtype=np.float32)}
                    
                    if debug_counter < 10: # 限制列印次數
                        print(f"\n--- 預覽樣本 {debug_counter + 1} (增強數據) ---")
                        print(f"圖片路徑: {img_path} (增強後)")
                        print(f"預處理後圖片形狀 (增強): {augmented_image_rgb_normalized.shape}")
                        print(f"**正規化感測器數據 (增強): {sensor_data_augmented_normed}**")
                        print(f"**正規化控制輸出 (轉向角 增強): {steering_output_augmented_normed}**")
                        print(f"**正規化控制輸出 (油門 增強): {throttle_output_augmented_normed}**")
                    debug_counter += 1
                else:
                    print(f"錯誤: 樣本 {img_path} 的增強感測器數據維度不匹配。預期 {config.SENSOR_INPUT_DIM}, 實際 {sensor_data_augmented_normed.shape[0]}。跳過此樣本。")
                    debug_counter += 1
                    continue # 跳過不符合維度的樣本
    
    # Modify by bun to cahnge output for dict.
    return tf.data.Dataset.from_generator(
        _generator,
        output_signature=(
            {
                'image_input': tf.TensorSpec(shape=(input_height, input_width, input_channels), dtype=tf.float32),  # 影像輸入
                'sensor_input': tf.TensorSpec(shape=(sensor_dim,), dtype=tf.float32)  # 感測器資料
            },
            # tf.TensorSpec(shape=(2,), dtype=tf.float32)  # 標籤（angle, throttle）
            {
                'steering_output': tf.TensorSpec(shape=(1,), dtype=tf.float32),
                'throttle_output': tf.TensorSpec(shape=(1,), dtype=tf.float32)
            }
        )
    )

    """
        # 整合感測器數據
    sensor_data = np.array(
        [rotation_x, rotation_y, rotation_Z] + \
        [angular_velocity_x, angular_velocity_y, angular_velocity_z] + \
        [speed, rollAngle, distFront, distRear, distLeft, distRight],
        dtype=np.float32
    )
    """

    """ read on first
    raw_angle_original = data[1]
    raw_throttle_original = data[2]
    raw_speed_original = data[3]
    raw_rotation_x_original, raw_rotation_y_original, raw_rotation_z_original = data[4]
    raw_angular_velocity_x_original, raw_angular_velocity_y_original, raw_angular_velocity_z_original = data[5]
    raw_rollAngle_original = data[6]
    raw_distFront_original = data[7]
    raw_distRear_original = data[8]
    raw_distLeft_original = data[9]
    raw_distRight_original = data[10]
    """
    """
    # 正規化原始感測器數據
    speed = config.norm_value(raw_speed_original, config.SENSOR_RANGES['speed']['min'], config.SENSOR_RANGES['speed']['max'])
    rotation_x = config.norm_value(raw_rotation_x_original, config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0)
    rotation_y = config.norm_value(raw_rotation_y_original, config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0)
    rotation_Z = config.norm_value(raw_rotation_z_original, config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0)
    angular_velocity_x = config.norm_value(raw_angular_velocity_x_original, config.SENSOR_RANGES['angular_velocity_x']['min'], config.SENSOR_RANGES['angular_velocity_x']['max'], target_min=-1.0, target_max=1.0)
    angular_velocity_y = config.norm_value(raw_angular_velocity_y_original, config.SENSOR_RANGES['angular_velocity_y']['min'], config.SENSOR_RANGES['angular_velocity_y']['max'], target_min=-1.0, target_max=1.0)
    angular_velocity_z = config.norm_value(raw_angular_velocity_z_original, config.SENSOR_RANGES['angular_velocity_z']['min'], config.SENSOR_RANGES['angular_velocity_z']['max'], target_min=-1.0, target_max=1.0)
    rollAngle = config.norm_value(raw_rollAngle_original, config.SENSOR_RANGES['rollAngle']['min'], config.SENSOR_RANGES['rollAngle']['max'], target_min=-1.0, target_max=1.0)
    distFront = config.norm_value(raw_distFront_original, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
    distRear = config.norm_value(raw_distRear_original, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
    distLeft = config.norm_value(raw_distLeft_original, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
    distRight = config.norm_value(raw_distRight_original, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
    """


    """
    # 複製原始圖片和數據，避免修改原始數據
    img_flipped = np.copy(img)
                
    # 複製原始數據，用於翻轉使用
    raw_angle_flipped = raw_angle_original
    raw_throttle_flipped = raw_throttle_original
    raw_speed_flipped = raw_speed_original
    raw_rotation_x_flipped, raw_rotation_y_flipped, raw_rotation_z_flipped = raw_rotation_x_original, raw_rotation_y_original, raw_rotation_z_original
    raw_angular_velocity_x_flipped, raw_angular_velocity_y_flipped, raw_angular_velocity_z_flipped = raw_angular_velocity_x_original, raw_angular_velocity_y_original, raw_angular_velocity_z_original
    raw_rollAngle_flipped = raw_rollAngle_original
    raw_distFront_flipped = raw_distFront_original
    raw_distRear_flipped = raw_distRear_original
    raw_distLeft_flipped = raw_distLeft_original
    raw_distRight_flipped = raw_distRight_original

    # 將圖像翻轉
    img_flipped = cv2.flip(img_flipped, 1)
    # 油門和速度保持不變
    # 角度翻轉
    raw_angle_flipped = -raw_angle_flipped
    # 旋轉角度調整 (Yaw 軸)
    raw_rotation_y_flipped = (360 - raw_rotation_y_flipped) % 360 # raw_rotation_y_flipped = -raw_rotation_y_flipped
    # 角速度調整 (Yaw Rate 軸)
    raw_angular_velocity_y_flipped = -raw_angular_velocity_y_flipped
    raw_angular_velocity_z_flipped = -raw_angular_velocity_z_flipped # 需測試是否需要翻轉
    # 翻轉角保持不變
    # 距離感測器左右互換
    temp_distLeft = raw_distLeft_flipped
    raw_distLeft_flipped = raw_distRight_flipped
    raw_distRight_flipped = temp_distLeft

    # --- 正規化翻轉後的數據 ---
    speed_flipped = config.norm_value(raw_speed_flipped, config.SENSOR_RANGES['speed']['min'], config.SENSOR_RANGES['speed']['max'])
    rotation_x_flipped = config.norm_value(raw_rotation_x_flipped, config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0)
    rotation_y_flipped = config.norm_value(raw_rotation_y_flipped, config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0)
    rotation_z_flipped = config.norm_value(raw_rotation_z_flipped, config.SENSOR_RANGES['rotation_xyz']['min'], config.SENSOR_RANGES['rotation_xyz']['max'], target_min=-1.0, target_max=1.0)
    angular_velocity_x_flipped = config.norm_value(raw_angular_velocity_x_flipped, config.SENSOR_RANGES['angular_velocity_x']['min'], config.SENSOR_RANGES['angular_velocity_x']['max'], target_min=-1.0, target_max=1.0)
    angular_velocity_y_flipped = config.norm_value(raw_angular_velocity_y_flipped, config.SENSOR_RANGES['angular_velocity_y']['min'], config.SENSOR_RANGES['angular_velocity_y']['max'], target_min=-1.0, target_max=1.0)
    angular_velocity_z_flipped = config.norm_value(raw_angular_velocity_z_flipped, config.SENSOR_RANGES['angular_velocity_z']['min'], config.SENSOR_RANGES['angular_velocity_z']['max'], target_min=-1.0, target_max=1.0)
    rollAngle_flipped = config.norm_value(raw_rollAngle_flipped, config.SENSOR_RANGES['rollAngle']['min'], config.SENSOR_RANGES['rollAngle']['max'], target_min=-1.0, target_max=1.0)
    distFront_flipped = config.norm_value(raw_distFront_flipped, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
    distRear_flipped = config.norm_value(raw_distRear_flipped, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
    distLeft_flipped = config.norm_value(raw_distLeft_flipped, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])
    distRight_flipped = config.norm_value(raw_distRight_flipped, config.SENSOR_RANGES['distance']['min'], config.SENSOR_RANGES['distance']['max'])

    sensor_data_flipped = np.array(
        [rotation_x_flipped, rotation_y_flipped, rotation_z_flipped] + \
        [angular_velocity_x_flipped, angular_velocity_y_flipped, angular_velocity_z_flipped] + \
        [speed_flipped, rollAngle_flipped, distFront_flipped, distRear_flipped, distLeft_flipped, distRight_flipped],
        dtype=np.float32
    )
    # Modify by bun to change output is 2 data
                                        
    #output_flipped = np.array([
    #    config.norm_value(raw_angle_flipped, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'], target_min=-1.0, target_max=1.0),
    #    config.norm_value(raw_throttle_flipped, config.LABEL_RANGES['throttle']['min'], config.LABEL_RANGES['throttle']['max'], target_min=0.0, target_max=1.0)
    #], dtype=np.float32)
                                        

    # 將輸出資料拆為兩個獨立的變數
    steering_output_flipped = config.norm_value(raw_angle_flipped, config.LABEL_RANGES['angle']['min'], config.LABEL_RANGES['angle']['max'], target_min=-1.0, target_max=1.0)
    throttle_output_flipped = config.norm_value(raw_throttle_flipped, config.LABEL_RANGES['throttle']['min'], config.LABEL_RANGES['throttle']['max'], target_min=0.0, target_max=1.0)
    """