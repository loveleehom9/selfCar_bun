import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from models import create_autodrive_model
import config

def preprocess_image(img_path, target_size=(634, 356)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"錯誤：無法讀取圖片 {img_path}")
        return None
    
    # 將圖片轉換為 HSV 處理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    
    # 儲存一份原始圖片，以供後續疊加熱力圖使用
    original_img = img.copy()

    img = np.expand_dims(img, axis=0)
    return img, original_img

def parse_sensor_data_from_log(log_path, target_timestamp):
    """
    從檔案中提取特定時間戳的 12 個感測器輸入數據。
    """
    if not os.path.exists(log_path):
        print(f"錯誤：找不到感測器數據檔案 {log_path}")
        return None

    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.startswith("StartDateTime") or line.startswith("Timestamp") or not line.strip():
                    continue
                
                parts = line.strip().split(';')
                if len(parts) < 3:
                    continue

                log_timestamp = f"{float(parts[0].strip()):.2f}"
                
                if log_timestamp == target_timestamp:
                    # 定義感測器的資料順序
                    sensor_order = [
                        "Velocity", "Rot_X", "Rot_Y", "Rot_Z", 
                        "AngularVelocity_X", "AngularVelocity_Y", "AngularVelocity_Z",
                        "rollAngle", "DistanceFront", "DistanceRear", 
                        "DistanceLeft", "DistanceRight"
                    ]

                    # 最後用來儲存正規化後的感測器數據
                    normed_sensor_dict = {}
                    
                    for part in parts:
                        part = part.strip()
                        if part.startswith("Velocity:"):
                            normed_sensor_dict["Velocity"] = float(part.replace("Velocity:", "").strip())
                        elif part.startswith("Rot:"):
                            values = [float(v) for v in part.replace("Rot: (", "").replace(")", "").strip().split(', ')]
                            normed_sensor_dict["Rot_X"] = values[0]
                            normed_sensor_dict["Rot_Y"] = values[1]
                            normed_sensor_dict["Rot_Z"] = values[2]
                        elif part.startswith("AngularVelocity:"):
                            values = [float(v) for v in part.replace("AngularVelocity: (", "").replace(")", "").strip().split(', ')]
                            normed_sensor_dict["AngularVelocity_X"] = values[0]
                            normed_sensor_dict["AngularVelocity_Y"] = values[1]
                            normed_sensor_dict["AngularVelocity_Z"] = values[2]
                        elif part.startswith("rollAngle:"):
                            normed_sensor_dict["rollAngle"] = float(part.replace("rollAngle:", "").strip())
                        elif part.startswith("DistanceFront:"):
                            normed_sensor_dict["DistanceFront"] = float(part.replace("DistanceFront:", "").strip())
                        elif part.startswith("DistanceRear:"):
                            normed_sensor_dict["DistanceRear"] = float(part.replace("DistanceRear:", "").strip())
                        elif part.startswith("DistanceLeft:"):
                            normed_sensor_dict["DistanceLeft"] = float(part.replace("DistanceLeft:", "").strip())
                        elif part.startswith("DistanceRight:"):
                            normed_sensor_dict["DistanceRight"] = float(part.replace("DistanceRight:", "").strip())

                    sensor_values = [normed_sensor_dict.get(name, 0.0) for name in sensor_order]
                    if len(sensor_values) == 12:
                        print(f"找到時間戳 {target_timestamp} 的感測器原始輸入數據：{sensor_values}")
                    
                    # 資料正規化處理
                    normed_sensor_dict["Velocity"] = config.norm_value(
                        normed_sensor_dict.get("Velocity", 0.0),
                        config.SENSOR_RANGES['speed']['min'], 
                        config.SENSOR_RANGES['speed']['max']
                    )
                    normed_sensor_dict["Rot_X"] = config.norm_value(
                        normed_sensor_dict.get("Rot_X", 0.0),
                        config.SENSOR_RANGES['rotation_xyz']['min'], 
                        config.SENSOR_RANGES['rotation_xyz']['max']
                    )
                    normed_sensor_dict["Rot_Y"] = config.norm_value(
                        normed_sensor_dict.get("Rot_Y", 0.0),
                        config.SENSOR_RANGES['rotation_xyz']['min'], 
                        config.SENSOR_RANGES['rotation_xyz']['max']
                    )
                    normed_sensor_dict["Rot_Z"] = config.norm_value(
                        normed_sensor_dict.get("Rot_Z", 0.0),
                        config.SENSOR_RANGES['rotation_xyz']['min'], 
                        config.SENSOR_RANGES['rotation_xyz']['max']
                    )
                    normed_sensor_dict["AngularVelocity_X"] = config.norm_value(
                        normed_sensor_dict.get("AngularVelocity_X", 0.0),
                        config.SENSOR_RANGES['angular_velocity_x']['min'], 
                        config.SENSOR_RANGES['angular_velocity_x']['max'], 
                        target_min=-1.0, target_max=1.0
                    )
                    normed_sensor_dict["AngularVelocity_Y"] = config.norm_value(
                        normed_sensor_dict.get("AngularVelocity_Y", 0.0),
                        config.SENSOR_RANGES['angular_velocity_y']['min'], 
                        config.SENSOR_RANGES['angular_velocity_y']['max'], 
                        target_min=-1.0, target_max=1.0
                    )
                    normed_sensor_dict["AngularVelocity_Z"] = config.norm_value(
                        normed_sensor_dict.get("AngularVelocity_Z", 0.0),
                        config.SENSOR_RANGES['angular_velocity_z']['min'], 
                        config.SENSOR_RANGES['angular_velocity_z']['max'], 
                        target_min=-1.0, target_max=1.0
                    )
                    normed_sensor_dict["rollAngle"] = config.norm_value(
                        normed_sensor_dict.get("rollAngle", 0.0),
                        config.SENSOR_RANGES['rollAngle']['min'], 
                        config.SENSOR_RANGES['rollAngle']['max'], 
                        target_min=-1.0, target_max=1.0
                    )
                    normed_sensor_dict["DistanceFront"] = config.norm_value(
                        normed_sensor_dict.get("DistanceFront", 0.0),
                        config.SENSOR_RANGES['distance']['min'], 
                        config.SENSOR_RANGES['distance']['max']
                    )
                    normed_sensor_dict["DistanceRear"] = config.norm_value(
                        normed_sensor_dict.get("DistanceRear", 0.0),
                        config.SENSOR_RANGES['distance']['min'], 
                        config.SENSOR_RANGES['distance']['max']
                    )
                    normed_sensor_dict["DistanceLeft"] = config.norm_value(
                        normed_sensor_dict.get("DistanceLeft", 0.0),
                        config.SENSOR_RANGES['distance']['min'], 
                        config.SENSOR_RANGES['distance']['max']
                    )
                    normed_sensor_dict["DistanceRight"] = config.norm_value(
                        normed_sensor_dict.get("DistanceRight", 0.0),
                        config.SENSOR_RANGES['distance']['min'], 
                        config.SENSOR_RANGES['distance']['max']
                    )
                    # 正規化結束

                    sensor_values = [normed_sensor_dict.get(name, 0.0) for name in sensor_order] 

                    if len(sensor_values) == 12:
                        print(f"找到時間戳 {target_timestamp} 的感測器輸入數據：{sensor_values}")
                        return np.array(sensor_values, dtype=np.float32).reshape(1, -1)
                    else:
                        print(f"警告：時間戳 {target_timestamp} 的感測器數據長度不正確 ({len(sensor_values)}/12)。")
                        return None
        
        print(f"錯誤：在檔案中找不到時間戳為 {target_timestamp} 的數據。")
        return None
    except Exception as e:
        print(f"錯誤：解析檔案時發生問題。詳細錯誤訊息：{e}")
        return None

def extract_timestamp_from_filename(img_path):
    """
    從圖像檔名中提取時間戳。
    e.g., 'Screenshot_0.00.jpg' -> '0.00'
    """
    filename = os.path.basename(img_path)
    if '_' in filename and '.' in filename:
        timestamp_str = filename.split('_')[-1].split('.jpg')[0]
        try:
            return f"{float(timestamp_str):.2f}"
        except ValueError:
            return None
    return None

# ----------------------------------------------------------------------
# 感測器重要性分析
# ----------------------------------------------------------------------
def get_sensor_importance(model, image_data, sensor_data, output_layer_name):
    """
    利用梯度計算每個感測器輸入對最終預測的重要性。
    """
    # 建立一個臨時模型，其輸出為指定的預測層
    sensor_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(output_layer_name).output
    )
    
    with tf.GradientTape() as tape:
        # 確保我們追蹤 sensor_data 的梯度
        tape.watch(sensor_data)
        # 進行預測
        prediction = sensor_model({'image_input': image_data, 'sensor_input': sensor_data})
        
    # 計算預測值對感測器數據的梯度
    grads = tape.gradient(prediction, sensor_data)
    
    # 梯度值代表了每個感測器對預測的影響程度
    return grads.numpy().flatten()

def get_gradcam_heatmap(model, image_array, last_conv_layer_name, output_layer_name, sensor_data):
    """
    為指定的模型輸出產生 Grad-CAM 熱力圖。
    """
    # 建立一個臨時模型，其輸出包括最後一個卷積層的特徵圖和最終的預測值
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.get_layer(output_layer_name).output]
    )

    with tf.GradientTape() as tape:
        # 使用 GradientTape 來追蹤梯度
        conv_output, preds = grad_model({'image_input': image_array, 'sensor_input': sensor_data})
        # 獲取最高預測值的索引
        pred_index = tf.argmax(preds[0])
        # 根據 pred_index 獲取目標輸出值（目前是轉向角跟油門值）
        target_output = preds[:, pred_index]

    # 計算目標輸出值對最後一個卷積層輸出特徵圖的梯度
    grads = tape.gradient(target_output, conv_output)

    # 梯度池化以獲得每個特徵圖的平均強度
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 乘以每個特徵圖的平均梯度，以獲得對預測有貢獻的區域
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 調整熱力圖的大小以匹配原始圖片大小
    heatmap = cv2.resize(heatmap.numpy(), (image_array.shape[2], image_array.shape[1]))
    
    # 正規化熱力圖
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap

# 梯度分析函數
def analyze_sensor_gradients(model, image_array, sensor_data, output_layer_names, sensor_feature_names):
    """
    分析模型輸出對於感測器輸入的梯度，找出感測器數據的重要性。

    Args:
        model: 訓練好的 TensorFlow Keras 模型。
        image_array: 預處理後的圖像輸入 (numpy array)。
        sensor_data: 預處理後的感測器輸入 (numpy array)。
        output_layer_names (list): 模型輸出層的名稱，例如 ["steering_output", "throttle_output"]。
        sensor_feature_names (list): 感測器數據的名稱列表

    Returns:
        dict: 包含每個輸出層對應的感測器梯度重要性排序。
    """
    print("\n正在分析感測器輸入的重要性 (梯度分析)...")
    
    # 將 numpy 數組轉換為 tf.Tensor，並設置 trainable=True 以便計算梯度
    tf_image_array = tf.constant(image_array, dtype=tf.float32)
    tf_sensor_data = tf.Variable(sensor_data, dtype=tf.float32) # sensor_data 必須是 tf.Variable 才能被追蹤梯度

    sensor_importance = {}

    for output_name in output_layer_names:
        with tf.GradientTape(persistent=True) as tape: # persistent=True 允許計算多個目標的梯度
            # 獲取模型輸出層
            output_tensor = model.get_layer(output_name).output
            # 建立一個臨時模型
            temp_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=output_tensor
            )
            # 執行前向傳播
            # 注意：這裡的 input 字典要和 create_autodrive_model 的 input 名稱一致
            predictions = temp_model({'image_input': tf_image_array, 'sensor_input': tf_sensor_data})
            
            # 使用 predictions[0] 來計算梯度
            target_output_value = predictions[0] 

        # 計算目前目標輸出對於感測器輸入的梯度
        grads = tape.gradient(target_output_value, tf_sensor_data)
        
        # 確保梯度不是 None
        if grads is None:
            print(f"警告：無法為 {output_name} 計算感測器梯度，可能是模型結構或圖未正確追蹤。")
            sensor_importance[output_name] = {}
            continue

        # 梯度通常是 (batch_size, num_sensors) 的形狀，我們取第一個樣本的梯度
        # 並且取絕對值，因為我們關心的是影響的“強度”
        sensor_grads = np.abs(grads.numpy().flatten())

        # 將感測器名稱和其梯度值配對
        importance_scores = {name: score for name, score in zip(sensor_feature_names, sensor_grads)}

        # 按照重要性分數降序排序
        sorted_importance = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
        sensor_importance[output_name] = sorted_importance
        
        print(f"--- {output_name} 的感測器重要性排序 ---\n")
        for feature, score in sorted_importance:
            print(f"  {feature}: {score:.6f}")
        
    return sensor_importance

def main():
    # 模型與圖片路徑
    model_path = './checkpoints/model_epoch_005.h5'
    img_path = 'C:/Users/User/source/repos/Car02/Cam01/20250728_184331/Screenshot_0.00.jpg'
    SENSOR_PATH = 'C:/Users/User/source/repos/Car02/Log/Log_20250728_184331.txt'

    check_single_data_flow(img_path, SENSOR_PATH)

    # 模型設定，根據你的模型摘要
    last_conv_layer_name = "conv5" # 抓取自己設定的 Conv 名稱
    output_layer_names = ["steering_output", "throttle_output"]

    # 定義感測器數據的順序和名稱
    sensor_feature_names = [
        "Velocity", "Rot_X", "Rot_Y", "Rot_Z", 
        "AngularVelocity_X", "AngularVelocity_Y", "AngularVelocity_Z",
        "rollAngle", "DistanceFront", "DistanceRear", 
        "DistanceLeft", "DistanceRight"
    ]

    # 載入模型
    #model = create_autodrive_model(input_shape=(356, 634, 3))
    model = create_autodrive_model(input_shape=(356, 634, 3), sensor_input_dim=12)
    model.load_weights(model_path)
    print("模型已載入")

    model.summary()

    # 圖片預處理
    img, original_img = preprocess_image(img_path)
    if img is None:
        return

    # 提取檔名中的時間戳
    target_timestamp = extract_timestamp_from_filename(img_path)
    if target_timestamp is None:
        print(f"錯誤：無法從圖片檔名 {img_path} 中提取時間戳。")
        return

    # 根據時間戳從檔案中獲取對應的感測器數據
    sensor_data = parse_sensor_data_from_log(SENSOR_PATH, target_timestamp)
    if sensor_data is None:
        return

    # 進行預測
    predictions = model.predict({'image_input': img, 'sensor_input': sensor_data})
    
    if len(predictions) != 2:
        print(f"錯誤：預測結果數量不正確。預期有 2 個輸出，但實際有 {len(predictions)} 個。")
        return

    # 由於我把輸出調整成獨立兩個變數，因此讀取上方式不同
    steering = float(predictions[0][0][0]) # float(predictions[0]) 
    throttle = float(predictions[1][0][0]) # float(predictions[1])
    print(f"預測結果 — 轉向角: {steering:.4f}; 油門: {throttle:.4f}")

    # --- GRAD-CAM 視覺化 ---
    print("\n正在生成 Grad-CAM 熱力圖...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # 1. 轉向角 (Steering) GRAD-CAM
    steering_heatmap = get_gradcam_heatmap(model, img, last_conv_layer_name, output_layer_names[0], sensor_data)
    steering_heatmap_colored = plt.get_cmap('jet')(steering_heatmap)[:, :, :3]
    steering_heatmap_colored = cv2.resize(steering_heatmap_colored.astype(np.float32), (original_img.shape[1], original_img.shape[0]))
    steering_superimposed = (steering_heatmap_colored * 0.4 + original_img * 0.6) * 255.0
    steering_superimposed = steering_superimposed.astype(np.uint8)

    # 2. 油門 (Throttle) GRAD-CAM
    throttle_heatmap = get_gradcam_heatmap(model, img, last_conv_layer_name, output_layer_names[1], sensor_data)
    throttle_heatmap_colored = plt.get_cmap('jet')(throttle_heatmap)[:, :, :3]
    throttle_heatmap_colored = cv2.resize(throttle_heatmap_colored.astype(np.float32), (original_img.shape[1], original_img.shape[0]))
    throttle_superimposed = (throttle_heatmap_colored * 0.4 + original_img * 0.6) * 255.0
    throttle_superimposed = throttle_superimposed.astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(cv2.cvtColor(steering_superimposed, cv2.COLOR_HSV2RGB))
    ax1.set_title(f"Steering Grad-CAM ({steering:.4f})")
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(throttle_superimposed, cv2.COLOR_HSV2RGB))
    ax2.set_title(f"Throttle Grad-CAM ({throttle:.4f})")
    ax2.axis('off')

    plt.suptitle("Grad-CAM for Steering And Throttle", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 儲存圖片
    plt.savefig("gradcam_combined_analysis.png")
    print("轉向角與油門 Grad-CAM 視覺化圖片已合併並儲存為 gradcam_combined_analysis.png")
    # plt.show()

    # --- 感測器重要性分析 ---
    sensor_importance_results = analyze_sensor_gradients(
        model, 
        img, 
        sensor_data, 
        output_layer_names, 
        sensor_feature_names
    )

    print("\n繪製合併的感測器重要性圖表...")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for i, (output_name, sorted_importance) in enumerate(sensor_importance_results.items()):
        if sorted_importance:
            features = [item for item, _ in sorted_importance]
            scores = [score for _, score in sorted_importance]

            ax = axes.flatten()[i] # 選擇當前子圖

            ax.barh(features, scores, color='lightcoral')
            ax.set_xlabel("importance")
            ax.set_ylabel("sensor name")
            ax.set_title(f"{output_name} sensor importance")
            ax.invert_yaxis() # 讓最重要的放在上面

    plt.suptitle("Sensor Importance", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig("combined_sensor_importance.png")
    print("合併的感測器重要性圖表已儲存為 combined_sensor_importance.png")
    # plt.show() 

# ======================================================================
# 新增的數據流程檢查函數
# ======================================================================
def check_single_data_flow(img_path, sensor_path):
    """
    檢查單一圖像和對應感測器數據的完整預處理流程。
    """
    print("--- 執行單一數據點的流程檢查 ---")
    
    # 1. 從檔名提取時間戳
    target_timestamp = extract_timestamp_from_filename(img_path)
    if target_timestamp is None:
        print("❌ 錯誤：無法從圖片檔名中提取時間戳，檢查終止。")
        return
    print(f"✅ 成功提取時間戳：{target_timestamp}")

    # 2. 預處理圖像
    img, original_img = preprocess_image(img_path)
    if img is None:
        print("❌ 錯誤：圖像預處理失敗，檢查終止。")
        return
    print(f"✅ 成功預處理圖像。圖像形狀: {img.shape}")
    
    # 3. 解析感測器數據
    sensor_data = parse_sensor_data_from_log(sensor_path, target_timestamp)
    if sensor_data is None:
        print("❌ 錯誤：感測器數據解析失敗，檢查終止。")
        return
    print(f"✅ 成功解析感測器數據。數據形狀: {sensor_data.shape}")
    
    # 4. 視覺化結果
    fig, ax = plt.subplots(figsize=(10, 5))
    # 將原始的 HSV 圖像轉換回 RGB 顯示
    img_display = cv2.cvtColor(original_img.astype(np.float32), cv2.COLOR_HSV2RGB)
    ax.imshow(img_display)
    ax.set_title(f"圖像樣本 - 時間戳: {target_timestamp}", fontsize=14)
    ax.axis('off')
    plt.show()

    # 5. 列印原始和正規化後的感測器數據
    print("\n--- 感測器數據總結 ---")
    
    # 注意：這裡的 `parse_sensor_data_from_log` 已經會印出原始數據
    # 您可以根據需要修改它來同時顯示正規化後的數據
    print(f"感測器正規化後的數值：{sensor_data.flatten()}")
    
    print("--- 檢查完成 ---")


if __name__ == '__main__':
    main()
