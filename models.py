import tensorflow as tf
from tensorflow.keras import layers, models, Input  
from tensorflow.keras.layers import Lambda , GlobalAveragePooling2D , MaxPooling2D#Add by bun 載入標準化及池化層測試
from tensorflow.keras import regularizers # 導入正則化器 Add by test nvidia cnn model
import config

# add by bun
# 設定寬度與高度及通道資訊
input_height = config.TARGET_IMAGE_HEIGHT
input_width = config.TARGET_IMAGE_WIDTH
input_channels = config.INPUT_CHANNELS
sensor_dim = config.SENSOR_INPUT_DIM

# 多模態自駕車模型的核心架構。
# 它結合了圖像的視覺資訊和感測器的數值資訊，用來預測車輛的轉向和油門。
def create_autodrive_model(input_shape=(input_height, input_width, input_channels),sensor_input_dim=sensor_dim):
    """
    多模態的自駕車模型，結合圖像與感測器數據。
    Args:
        input_shape (tuple): 圖像輸入的形狀，例如 (356, 634, 3)。
        sensor_input_dim (int): 感測器數據的維度。

    Returns:
        tf.keras.Model: 創建好的 Keras Functional API 模型。
    """


    """
    - 緩慢的向左撞，走得更遠
    - 使用解析度圖像為 (634x356)。
    - 使用步幅卷積 (strides=2)
    - 在卷積層和全連接層後使用 BatchNormalization。
    - 在 CNN Flatten 後添加 Dense 層壓縮特徵。
    - 使用 ELU 激活函數和 Dropout。
    """

    # --- 1. 圖像輸入與 CNN 特徵提取分支 ---
    #  Input 層專門接收圖像數據。
    image_input = Input(shape=input_shape, name='image_input')

    # 標準化影像 將像素值從 [0, 255] 轉換到 [0, 1]。
    x = Lambda(lambda img: img / 255.0, name='normalization')(image_input)

    # 模仿 NVIDIA DAVE-2 的卷積神經網絡結構，
    # 使用步幅卷積(strides=2)來逐步縮小特徵圖尺寸，同時提取空間特徵。
    # 每個 Conv2D 後都接著 BatchNormalization 和 ELU 激活函數，深度學習的標準優化組合。

    # 第一卷積組：Conv2D (strides=2) -> BatchNormalization -> ELU Activation
    # 輸出尺寸計算: ((輸入尺寸 - 濾波器尺寸) / 步幅) + 1
    # H = ((356 - 5) / 2) + 1 = 176.5 -> 向下取整為 176
    # W = ((634 - 5) / 2) + 1 = 315.5 -> 向下取整為 315
    # Conv Block 1: 32個濾波器，5x5內核，步幅2
    #x = layers.Conv2D(32, (5, 5), strides=2, padding='valid', name='conv1')(image_input) # 移除 , activation='linear'
    # test to lambda
    x = layers.Conv2D(32, (5, 5), strides=2, padding='valid', name='conv1')(x) # 移除 , activation='linear'
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('elu', name='elu1')(x)
    # 當前特徵圖尺寸: 32@176x315

    # 第二卷積組：Conv2D (strides=2) -> BatchNormalization -> ELU Activation
    # H = ((176 - 3) / 2) + 1 = 87.5 -> 向下取整為 87
    # W = ((315 - 3) / 2) + 1 = 157.5 -> 向下取整為 157
    # Conv Block 2: 64個濾波器，3x3內核，步幅2
    x = layers.Conv2D(64, (3, 3), strides=2, padding='valid', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('elu', name='elu2')(x)
    # 當前特徵圖尺寸: 64@87x157

    # 第三卷積組：Conv2D (strides=2) -> BatchNormalization -> ELU Activation
    # H = ((87 - 3) / 2) + 1 = 43.5 -> 向下取整為 43
    # W = ((157 - 3) / 2) + 1 = 77.5 -> 向下取整為 77
    # Conv Block 3: 128個濾波器，3x3內核，步幅2
    x = layers.Conv2D(128, (3, 3), strides=2, padding='valid', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('elu', name='elu3')(x)
    # 當前特徵圖尺寸: 128@43x77 (目前三次 strides=2 的下採樣，總下採樣因子為 8)

    # 第四卷積層：Conv2D (strides=1) -> BatchNormalization -> ELU Activation
    # H = ((43 - 3) / 1) + 1 = 41
    # W = ((77 - 3) / 1) + 1 = 75
    # Conv Block 4: 64個濾波器，3x3內核，步幅1 (不縮小尺寸)
    x = layers.Conv2D(64, (3, 3), strides=1, padding='valid' , name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('elu', name='elu4')(x)
    # 當前特徵圖尺寸: 64@41x75

    # 第五卷積層：Conv2D (strides=1) -> BatchNormalization -> ELU Activation
    # H = ((41 - 3) / 1) + 1 = 39
    # W = ((75 - 3) / 1) + 1 = 73
    # Conv Block 5: 64個濾波器，3x3內核，步幅1
    x = layers.Conv2D(64, (3, 3), strides=1, padding='valid' , name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.Activation('elu', name='elu5')(x)
    # 當前特徵圖尺寸: 64@39x73
    
    # Mark by bun 減少參數數量嘗試
    """
    # Flatten CNN
    # Flatten 層將 2D 的特徵圖轉換成 1D 的向量，以便傳遞給全連接層。
    x = layers.Flatten(name='flatten')(x)
    # Flatten 輸出維度計算: 64 * 39 * 73 = 182,304 個神經元
    
    # 新增於 Flatten 後添加 Dense 層來壓縮圖像特徵
    # Dense 層將龐大的圖像特徵向量壓縮到 256 維，減少計算量並避免過擬合。
    image_features = layers.Dense(256, name='cnn_features_dense')(x) # 壓縮到 256 維
    image_features = layers.BatchNormalization(name='bn_cnn_features')(image_features)
    image_features = layers.Activation('elu', name='elu_cnn_features')(image_features)
    # Dropout 層隨機關閉一部分神經元，進一步防止過擬合。
    image_features = layers.Dropout(0.5, name='dropout_cnn_features')(image_features) # 在壓縮後應用 Dropout
    """
    # Add by bun
    # image_features = GlobalAveragePooling2D(name='global_average_pooling')(x)

    # Modify by bun to UNITY CHANGE

    # Mark by bun 減少參數數量嘗試
    # 這裡將 GlobalAveragePooling2D 替換為 Flatten 和 Dense 層，以解決 ONNX 轉換問題
    # Flatten 層將 2D 的特徵圖轉換成 1D 的向量，以便傳遞給全連接層。
    # x = layers.Flatten(name='flatten')(x)
    # Flatten 輸出維度計算: 64 * 39 * 73 = 182,304 個神經元

    """
    # 新增於 Flatten 後添加 Dense 層來壓縮圖像特徵
    # Dense 層將龐大的圖像特徵向量壓縮到 256 維，減少計算量並避免過擬合。
    image_features = layers.Dense(256, name='cnn_features_dense')(x) # 壓縮到 256 維
    image_features = layers.BatchNormalization(name='bn_cnn_features')(image_features)
    image_features = layers.Activation('elu', name='elu_cnn_features')(image_features)
    # Dropout 層隨機關閉一部分神經元，進一步防止過擬合。
    image_features = layers.Dropout(0.5, name='dropout_cnn_features')(image_features) # 在壓縮後應用 Dropout
    """

    # Try ONNX used. 測試ONNX，是否可以正常運作
    # **新增**：在 Flatten 之前，先使用 MaxPooling2D 對特徵圖進行下採樣。
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # **新增**：Flatten 層將 2D 特徵圖轉換成 1D 向量。
    x = layers.Flatten(name='flatten')(x)

    # **新增**：新增 Dense 層來壓縮圖像特徵。
    image_features = layers.Dense(256, name='cnn_features_dense')(x)
    image_features = layers.BatchNormalization(name='bn_cnn_features')(image_features)
    image_features = layers.Activation('elu', name='elu_cnn_features')(image_features)
    image_features = layers.Dropout(0.5, name='dropout_cnn_features')(image_features)
    

    # --- 2. 感測器輸入處理分支 (全連接層) ---
    # Input 層專門接收來自感測器的數值數據。
    sensor_input = Input(shape=(sensor_input_dim,), name='sensor_input')

    # Dense 層處理感測器數據，將其轉換為更抽象的特徵。
    # 使用 ReLU 激活函數，應用在處理數值型數據時表現良好。
    s = layers.Dense(64, name='sensor_dense1')(sensor_input) # 增加神經元數量
    s = layers.BatchNormalization(name='bn_sensor1')(s)
    s = layers.Activation('relu', name='relu_sensor1')(s)

    s = layers.Dense(32, name='sensor_dense2')(s) # 增加神經元數量
    s = layers.BatchNormalization(name='bn_sensor2')(s)
    s = layers.Activation('relu', name='relu_sensor2')(s)

    sensor_features = layers.Dropout(0.3, name='dropout_sensor_features')(s) # 在感測器特徵後應用 Dropout


    # --- 3. 合併 CNN 與感測器特徵 ---
    # 將壓縮後的圖像特徵和感測器特徵拼接成一個單一的向量。
    combined = layers.Concatenate(name='concatenate_features')([image_features, sensor_features])

    # --- 4. 全連接層輸出 angle 和 throttle ---
    # 增加全連接層的神經元數量，並加入 BatchNormalization 和 Dropout
    # 從合併的特徵中學習高階的駕駛決策。
    fc = layers.Dense(256, name='fc1')(combined) # 增加神經元數量
    fc = layers.BatchNormalization(name='bn_fc1')(fc)
    fc = layers.Activation('elu', name='elu_fc1')(fc)

    # Modify by bun use 0.5 to 0.7
    # 這是為了更強地隨機失活神經元，避免模型過度依賴訓練數據中的特定模式。
    fc = layers.Dropout(0.7, name='dropout_fc1')(fc) # 增加 Dropout

    # Modify by bun use 128 to 32
    fc = layers.Dense(32, name='fc2')(fc) # 增加神經元數量

    fc = layers.BatchNormalization(name='bn_fc2')(fc)
    fc = layers.Activation('elu', name='elu_fc2')(fc)
    fc = layers.Dropout(0.5, name='dropout_fc2')(fc) # 增加 Dropout

    # Mark by bun to test 縮減參數
    """
    fc = layers.Dense(64, name='fc3')(fc) # 增加神經元數量
    fc = layers.BatchNormalization(name='bn_fc3')(fc)
    fc = layers.Activation('elu', name='elu_fc3')(fc)
    fc = layers.Dropout(0.3, name='dropout_fc3')(fc) # 增加 Dropout
    """

    # Mark by bun
    # output = layers.Dense(2, name='control_output')(fc)
    
    # 最後輸出 (轉向角, 油門)
    # 最後的 Dense 層是模型的輸出層，它有 2 個神經元，分別對應轉向角和油門。
    # 這裡使用線性激活，讓模型直接輸出數值，損失函數會負責優化這些輸出。
    # Mark by bun test output Data
    # output = layers.Dense(2, name='control_output')(fc)
    # Add by bun to test two output data
    # x = layers.Dense(64, activation='elu')(fc)

    # 轉向角輸出層，使用 tanh 激活函數，範圍 [-1, 1]
    # steering_output = layers.Dense(1, activation='tanh', name='steering_output')(x)
    # 油門輸出層，使用 sigmoid 激活函數，範圍 [0, 1]
    # throttle_output = layers.Dense(1, activation='sigmoid', name='throttle_output')(x)
    # 將兩個輸出合併
    # output = layers.concatenate([steering_output, throttle_output])
    # 專屬註解：Model 函數將輸入和輸出連接起來，打造出完整的模型。
    #model = models.Model(inputs=[image_input, sensor_input], outputs=output)

    # 預測轉向角的輸出層。
    # 使用 tanh 激活函數，將輸出範圍限制在 [-1, 1]。
    steering_output = layers.Dense(1, activation='tanh', name='steering_output')(fc)
    
    # 預測油門的輸出層。
    # 使用 sigmoid 激活函數，將輸出範圍限制在 [0, 1]。
    throttle_output = layers.Dense(1, activation='sigmoid', name='throttle_output')(fc)
    
    # Model 函數將兩個輸入和兩個輸出連接起來，構建出完整的模型。
    # 「多輸出模型」，每個輸出都可以有獨立的損失函數。
    model = models.Model(inputs=[image_input, sensor_input], outputs=[steering_output, throttle_output])


    # Mark by bun to combin in main test
    """
    # Add by bun to see model data
    # 打印模型摘要，查看各層的輸出形狀和參數數量
    model.summary()
    # 簡單測試模型輸入和輸出形狀
    import numpy as np
    dummy_image = np.random.rand(1, config.TARGET_IMAGE_HEIGHT, config.TARGET_IMAGE_WIDTH, 3).astype(np.float32)
    dummy_sensor_data = np.random.rand(1, config.SENSOR_INPUT_DIM).astype(np.float32)
    dummy_output = model({'image_input': dummy_image, 'sensor_input': dummy_sensor_data})
    print(f"\n模型輸出形狀: {dummy_output.shape}") # 預期 (1, 2)
    """
    return model

# 簡單的測試，用於驗證模型是否能正確建立和運行。
if __name__ == '__main__':
    TARGET_IMAGE_HEIGHT = config.TARGET_IMAGE_HEIGHT
    TARGET_IMAGE_WIDTH = config.TARGET_IMAGE_WIDTH
    INPUT_CHANNELS = config.INPUT_CHANNELS
    SENSOR_INPUT_DIM = config.SENSOR_INPUT_DIM

    dummy_model = create_autodrive_model(
        input_shape=(config.TARGET_IMAGE_HEIGHT, config.TARGET_IMAGE_WIDTH, config.INPUT_CHANNELS),
        sensor_input_dim=config.SENSOR_INPUT_DIM
    )
    
    print("模型摘要:")
    dummy_model.summary()

    import numpy as np
    dummy_image = np.random.rand(1, config.TARGET_IMAGE_HEIGHT, config.TARGET_IMAGE_WIDTH, 3).astype(np.float32)
    dummy_sensor_data = np.random.rand(1, config.SENSOR_INPUT_DIM).astype(np.float32)
    dummy_output = dummy_model({'image_input': dummy_image, 'sensor_input': dummy_sensor_data})
    
    # print(f"\n模型輸出形狀: {dummy_output.shape}") # 預期 (1, 2)
    # Modiy by bun for output check
    print(f"\n修正後的模型輸出形狀: {[o.shape for o in dummy_output]}")

############################################
### 目前使用過的模型 #######################
############################################

### 5 僅圖片訓練模組 ###
"""
def create_autodrive_model(input_shape=(input_height, input_width, input_channels)):
    
    純圖片自駕車模型，僅使用圖像數據來預測轉向角和油門。
    此版本移除了感測器輸入的分支。

    Args:
        input_shape (tuple): 圖像輸入的形狀，例如 (356, 634, 3)。

    Returns:
        tf.keras.Model: 創建好的 Keras Functional API 模型。
    

    # --- 1. 圖像輸入與 CNN 特徵提取分支 ---
    image_input = layers.Input(shape=input_shape, name='image_input')

    # 標準化影像
    x = layers.Lambda(lambda img: img / 255.0, name='normalization')(image_input)

    # 模仿 NVIDIA DAVE-2 的卷積神經網絡結構
    x = layers.Conv2D(32, (5, 5), strides=2, padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('elu', name='elu1')(x)

    x = layers.Conv2D(64, (3, 3), strides=2, padding='valid', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('elu', name='elu2')(x)

    x = layers.Conv2D(128, (3, 3), strides=2, padding='valid', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('elu', name='elu3')(x)

    x = layers.Conv2D(64, (3, 3), strides=1, padding='valid', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('elu', name='elu4')(x)

    x = layers.Conv2D(64, (3, 3), strides=1, padding='valid', name='conv5')(x)
    x = layers.BatchNormalization(name='bn5')(x)
    x = layers.Activation('elu', name='elu5')(x)
    
    # 這裡使用 GlobalAveragePooling2D 減少參數
    image_features = layers.GlobalAveragePooling2D(name='global_average_pooling')(x)

    # --- 2. 全連接層輸出 angle 和 throttle (直接從圖像特徵) ---
    fc = layers.Dense(256, name='fc1')(image_features)
    fc = layers.BatchNormalization(name='bn_fc1')(fc)
    fc = layers.Activation('elu', name='elu_fc1')(fc)
    fc = layers.Dropout(0.7, name='dropout_fc1')(fc)

    fc = layers.Dense(128, name='fc2')(fc)
    fc = layers.BatchNormalization(name='bn_fc2')(fc)
    fc = layers.Activation('elu', name='elu_fc2')(fc)
    fc = layers.Dropout(0.5, name='dropout_fc2')(fc)

    fc = layers.Dense(64, name='fc3')(fc)
    fc = layers.BatchNormalization(name='bn_fc3')(fc)
    fc = layers.Activation('elu', name='elu_fc3')(fc)
    fc = layers.Dropout(0.3, name='dropout_fc3')(fc)
    
    # 預測轉向角的輸出層。
    steering_output = layers.Dense(1, activation='tanh', name='steering_output')(fc)
    
    # 預測油門的輸出層。
    throttle_output = layers.Dense(1, activation='sigmoid', name='throttle_output')(fc)
    
    # Model 函數將輸入和輸出連接起來，構建出完整的模型。
    model = models.Model(inputs=[image_input], outputs=[steering_output, throttle_output])

    return model
"""
"""
#### 4 模仿輝達模組 ####
    
RELU模型 測試效果不佳
"""
"""
模擬 NVIDIA DAVE-2 核心卷積結構，並加入多模態輸入和多輸出，
整合 BatchNormalization 和 L2 正則化的 Keras Functional API 模型。

Args:
    input_shape (tuple): 圖像輸入形狀 (高, 寬, 通道)。
    sensor_input_dim (int): 感測器數據維度。
    l2_reg_const (float): L2 正則化常數。
"""
"""
l2_reg_const = 0.0001

# --- 圖像輸入與 CNN 特徵提取分支 (基於 NVIDIA DAVE-2) ---
image_input = Input(shape=input_shape, name='image_input')

# 第一卷積層: 5x5, 24 filters, strides=2 (下採樣)
# 原始 DAVE-2 使用 ReLU，這裡使用 BatchNormalization 後接 ReLU
x = layers.Conv2D(24, (5, 5), strides=2, padding='valid', activation='linear', 
                    kernel_regularizer=regularizers.l2(l2_reg_const), name='conv1')(image_input)
x = layers.BatchNormalization(name='bn1')(x)
x = layers.Activation('elu', name='elu1')(x) # 沿用 DAVE-2 的 ReLU

# 第二卷積層: 5x5, 36 filters, strides=2 (下採樣)
x = layers.Conv2D(36, (5, 5), strides=2, padding='valid', activation='linear',
                    kernel_regularizer=regularizers.l2(l2_reg_const), name='conv2')(x)
x = layers.BatchNormalization(name='bn2')(x)
x = layers.Activation('elu', name='elu2')(x)

# 第三卷積層: 5x5, 48 filters, strides=2 (下採樣)
x = layers.Conv2D(48, (5, 5), strides=2, padding='valid', activation='linear',
                    kernel_regularizer=regularizers.l2(l2_reg_const), name='conv3')(x)
x = layers.BatchNormalization(name='bn3')(x)
x = layers.Activation('elu', name='elu3')(x)

# 第四卷積層: 3x3, 64 filters, strides=1 (無下採樣)
x = layers.Conv2D(64, (3, 3), strides=1, padding='valid', activation='linear',
                    kernel_regularizer=regularizers.l2(l2_reg_const), name='conv4')(x)
x = layers.BatchNormalization(name='bn4')(x)
x = layers.Activation('elu', name='elu4')(x)

# 第五卷積層: 3x3, 64 filters, strides=1 (無下採樣)
x = layers.Conv2D(64, (3, 3), strides=1, padding='valid', activation='linear',
                    kernel_regularizer=regularizers.l2(l2_reg_const), name='conv5')(x)
x = layers.BatchNormalization(name='bn5')(x)
x = layers.Activation('elu', name='elu5')(x)
    
# 展平 CNN 輸出
x = layers.Flatten(name='flatten')(x)

# 在 Flatten 後添加 Dense 層壓縮圖像特徵，並加入 BatchNormalization 和 Dropout
# 壓縮到 256 維，這將是融合前圖像分支的輸出維度
image_features = layers.Dense(256, activation='linear', kernel_regularizer=regularizers.l2(l2_reg_const), name='cnn_features_dense')(x)
image_features = layers.BatchNormalization(name='bn_cnn_features')(image_features)
image_features = layers.Activation('elu', name='elu_cnn_features')(image_features) # 沿用 ReLU
image_features = layers.Dropout(0.5, name='dropout_cnn_features')(image_features)

# --- 感測器輸入處理分支 ---
sensor_input = Input(shape=(sensor_input_dim,), name='sensor_input')
# 增加神經元數量，並加入 BatchNormalization 和 ReLU 激活
s = layers.Dense(64, activation='linear', kernel_regularizer=regularizers.l2(l2_reg_const), name='sensor_dense1')(sensor_input)
s = layers.BatchNormalization(name='bn_sensor1')(s)
s = layers.Activation('elu', name='elu_sensor1')(s)
s = layers.Dense(32, activation='linear', kernel_regularizer=regularizers.l2(l2_reg_const), name='sensor_dense2')(s)
s = layers.BatchNormalization(name='bn_sensor2')(s)
s = layers.Activation('elu', name='elu_sensor2')(s)
sensor_features = layers.Dropout(0.3, name='dropout_sensor_features')(s)

# --- 合併 CNN 與感測器特徵 ---
# 將壓縮後的圖像特徵和感測器特徵拼接起來
combined = layers.Concatenate(name='concatenate_features')([image_features, sensor_features])

# --- 全連接層輸出 (基於 DAVE-2 的 FC 層結構，並調整尺寸和加入最佳實踐) ---
# DAVE-2 原始 FC 層神經元數量: 1164 -> 100 -> 50 -> 10
# 這裡調整為更適合融合後特徵的尺寸，並加入 BatchNormalization 和 Dropout
    
# FCL 1 (類似 DAVE-2 的第一層 FC)
fc = layers.Dense(256, activation='linear', kernel_regularizer=regularizers.l2(l2_reg_const), name='fc1')(combined)
fc = layers.BatchNormalization(name='bn_fc1')(fc)
fc = layers.Activation('elu', name='elu_fc1')(fc)
fc = layers.Dropout(0.5, name='dropout_fc1')(fc)

# FCL 2 (類似 DAVE-2 的第二層 FC)
fc = layers.Dense(128, activation='linear', kernel_regularizer=regularizers.l2(l2_reg_const), name='fc2')(fc)
fc = layers.BatchNormalization(name='bn_fc2')(fc)
fc = layers.Activation('elu', name='elu_fc2')(fc)
fc = layers.Dropout(0.5, name='dropout_fc2')(fc)

# FCL 3 (類似 DAVE-2 的第三層 FC)
fc = layers.Dense(64, activation='linear', kernel_regularizer=regularizers.l2(l2_reg_const), name='fc3')(fc)
fc = layers.BatchNormalization(name='bn_fc3')(fc)
fc = layers.Activation('elu', name='elu_fc3')(fc)
fc = layers.Dropout(0.3, name='dropout_fc3')(fc)

# FCL 4 (類似 DAVE-2 的第四層 FC)
fc = layers.Dense(32, activation='linear', kernel_regularizer=regularizers.l2(l2_reg_const), name='fc4')(fc)
fc = layers.BatchNormalization(name='bn_fc4')(fc)
fc = layers.Activation('elu', name='elu_fc4')(fc)
fc = layers.Dropout(0.3, name='dropout_fc4')(fc)

# 最終輸出層 (2 個輸出: 轉向角, 油門)
# 原始 DAVE-2 的轉向角輸出使用了 atan 激活並縮放。
# 這裡保持線性激活，讓損失函數直接優化歸一化後的輸出，
# 這與您目前訓練和反歸一化的流程一致，避免引入新的複雜性。
output = layers.Dense(2, name='control_output', kernel_regularizer=regularizers.l2(l2_reg_const))(fc) 


"""

#### 3 bun ####

"""
- 緩慢的向左撞
- 使用解析度圖像為 (634x356)。
:param input_shape: 圖像輸入大小（height, width, channels）
:sensor_input_dim: 感測器資料維度（如 raycast 4 + speed 1 + rotation 3 + 傾斜角度 2）
:return: Keras 模型物件
:輸出 angle 與 throttle
"""
""" Mark by test new nodel
# --- 1. 圖像輸入與 CNN 特徵提取分支 ---
image_input = Input(shape=input_shape, name='image_input')

# 第一卷積組：Conv2D (strides=2 進行下採樣) -> BatchNormalization -> ELU Activation
# 輸出尺寸計算: ((輸入尺寸 - 濾波器尺寸) / 步幅) + 1
# H = ((356 - 5) / 2) + 1 = 176.5 -> 向下取整為 176
# W = ((634 - 5) / 2) + 1 = 315.5 -> 向下取整為 315
x = layers.Conv2D(32, (5, 5), strides=2, padding='valid', activation='linear', name='conv1')(image_input)
x = layers.BatchNormalization(name='bn1')(x)
x = layers.Activation('elu', name='elu1')(x)
# 當前特徵圖尺寸: 32@176x315

# 第二卷積組：Conv2D (strides=2 進行下採樣) -> BatchNormalization -> ELU Activation
# H = ((176 - 3) / 2) + 1 = 87.5 -> 向下取整為 87
# W = ((315 - 3) / 2) + 1 = 157.5 -> 向下取整為 157
x = layers.Conv2D(64, (3, 3), strides=2, padding='valid', activation='linear', name='conv2')(x)
x = layers.BatchNormalization(name='bn2')(x)
x = layers.Activation('elu', name='elu2')(x)
# 當前特徵圖尺寸: 64@87x157

# 第三卷積組：Conv2D (strides=2 進行下採樣) -> BatchNormalization -> ELU Activation
# H = ((87 - 3) / 2) + 1 = 43.5 -> 向下取整為 43
# W = ((157 - 3) / 2) + 1 = 77.5 -> 向下取整為 77
x = layers.Conv2D(128, (3, 3), strides=2, padding='valid', activation='linear', name='conv3')(x)
x = layers.BatchNormalization(name='bn3')(x)
x = layers.Activation('elu', name='elu3')(x)
# 當前特徵圖尺寸: 128@43x77 (經過三次 strides=2 的下採樣，總下採樣因子為 8)

# 第四卷積層：Conv2D (strides=1，不再下採樣) -> ELU Activation
# H = ((43 - 3) / 1) + 1 = 41
# W = ((77 - 3) / 1) + 1 = 75
x = layers.Conv2D(64, (3, 3), strides=1, padding='valid', activation='elu', name='conv4')(x)
# 當前特徵圖尺寸: 64@41x75

# 第五卷積層：Conv2D (strides=1，不再下採樣) -> ELU Activation
# H = ((41 - 3) / 1) + 1 = 39
# W = ((75 - 3) / 1) + 1 = 73
x = layers.Conv2D(64, (3, 3), strides=1, padding='valid', activation='elu', name='conv5')(x)
# 當前特徵圖尺寸: 64@39x73
    
# Flatten 與 Dropout
x = layers.Flatten(name='flatten')(x)
# Flatten 輸出維度計算: 64 * 39 * 73 = 182,304 個神經元
x = layers.Dropout(0.5, name='dropout_cnn')(x)

# --- 2. 感測器輸入處理分支 ---
sensor_input = Input(shape=(sensor_input_dim,), name='sensor_input')
s = layers.Dense(32, activation='relu', name='sensor_dense1')(sensor_input)
s = layers.Dense(16, activation='relu', name='sensor_dense2')(s)

# --- 3. 合併 CNN 與感測器特徵 ---
combined = layers.Concatenate(name='concatenate_features')([x, s])

# --- 4. 全連接層輸出 angle 和 throttle ---
fc = layers.Dense(100, activation='elu', name='fc1')(combined)
fc = layers.Dropout(0.3, name='dropout_fc1')(fc)
fc = layers.Dense(50, activation='elu', name='fc2')(fc)
fc = layers.Dropout(0.3, name='dropout_fc2')(fc)
fc = layers.Dense(10, activation='elu', name='fc3')(fc)
output = layers.Dense(2, name='control_output')(fc) # 2 outputs: angle, throttle
Mark end by 0729
"""

#### 2 bun ####
"""
- 快速的向左撞
- 使用解析度圖像為 (634x356)。
:param input_shape: 圖像輸入大小（height, width, channels）
:sensor_input_dim: 感測器資料維度（如 raycast 4 + speed 1 + rotation 3 + 傾斜角度 2）
:return: Keras 模型物件
:輸出 angle 與 throttle
"""
""" Mark by bun for test 634*356
# --- 1. 圖像輸入與 CNN 特徵提取 ---
image_input = Input(shape=input_shape, name='image_input')
# 第一組：Conv2D -> BatchNormalization -> ELU Activation -> MaxPooling2D
# strides 設為 1，讓 MaxPooling2D 來做空間維度縮減
x = layers.Conv2D(32, (5, 5), strides=1, padding='valid', activation='linear')(image_input)
x = layers.BatchNormalization()(x) # 正規化
x = layers.Activation('elu')(x)    # 激活函數
x = layers.MaxPooling2D((2, 2))(x) # 池化層 (將尺寸減半)

# 第二組：
x = layers.Conv2D(64, (3, 3), strides=1, padding='valid', activation='linear')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('elu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# 第三組：
x = layers.Conv2D(128, (3, 3), strides=1, padding='valid', activation='linear')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('elu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='elu', padding='valid')(x)
x = layers.Conv2D(64, (3, 3), activation='elu', padding='valid')(x)
    
# Flatten 與 Dropout
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x) # 保持原有的 Dropout

# --- 2. 感測器輸入處理 ---
sensor_input = Input(shape=(sensor_input_dim,), name='sensor_input')
s = layers.Dense(32, activation='relu')(sensor_input)
s = layers.Dense(16, activation='relu')(s)

# --- 3. 合併 CNN 與感測器特徵 ---
combined = layers.Concatenate()([x, s])

# --- 4. 全連接層輸出 angle 和 throttle ---
fc = layers.Dense(100, activation='elu')(combined)
fc = layers.Dropout(0.3)(fc) # **新增 Dropout**
fc = layers.Dense(50, activation='elu')(fc)
fc = layers.Dropout(0.3)(fc) # **新增 Dropout**
fc = layers.Dense(10, activation='elu')(fc)
output = layers.Dense(2, name='control_output')(fc)
Mark End
"""

#### 4 Clary Lin ####
# Mark by bun
# 原始MODEL 向右撞
"""
# 圖像輸入與 CNN 特徵提取
image_input = Input(shape=input_shape, name='image_input')
x = layers.Conv2D(24, (5, 5), strides=2, activation='elu')(image_input)
x = layers.Conv2D(36, (5, 5), strides=2, activation='elu')(x)
x = layers.Conv2D(48, (5, 5), strides=2, activation='elu')(x)
x = layers.Conv2D(64, (3, 3), activation='elu')(x)
x = layers.Conv2D(64, (3, 3), activation='elu')(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)

# 感測器輸入處理
sensor_input = Input(shape=(sensor_input_dim,), name='sensor_input')
s = layers.Dense(32, activation='relu')(sensor_input)
s = layers.Dense(16, activation='relu')(s)

# 合併 CNN 與感測器特徵
combined = layers.Concatenate()([x, s])

# 全連接層輸出 angle 和 throttle
fc = layers.Dense(100, activation='elu')(combined)
fc = layers.Dense(50, activation='elu')(fc)
fc = layers.Dense(10, activation='elu')(fc)
output = layers.Dense(2, name='control_output')(fc)

# Mark End by bun
"""
