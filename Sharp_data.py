import tensorflow as tf
import numpy as np
import shap
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 步驟 1: 定義你的模型架構 (使用範例模型)
#
# 由於我無法存取你實際的 models.py 檔案，這裡創建一個與你描述相符的範例模型。
# 這個模型同時接受圖像和感測器數據作為輸入。
# -----------------------------------------------------------------------------
def create_autodrive_model(input_shape=(356, 634, 3), sensor_input_dim=12):
    """
    創建一個多輸入的自動駕駛模型，包含圖像和感測器輸入。
    """
    # 圖像輸入分支 (與你 Grad-CAM 程式碼中的設定類似)
    image_input = tf.keras.Input(shape=input_shape, name='image_input')
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv5')(x) # 假設這是你 Grad-CAM 的最後一個卷積層
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # 感測器輸入分支
    sensor_input = tf.keras.Input(shape=(sensor_input_dim,), name='sensor_input')
    y = tf.keras.layers.Dense(32, activation='relu')(sensor_input)

    # 將兩個分支的輸出合併
    combined = tf.keras.layers.concatenate([x, y])
    combined = tf.keras.layers.Dense(32, activation='relu')(combined)

    # 輸出層，兩個獨立的輸出 (轉向角和油門)
    steering_output = tf.keras.layers.Dense(1, activation='linear', name='steering_output')(combined)
    throttle_output = tf.keras.layers.Dense(1, activation='linear', name='throttle_output')(combined)

    model = tf.keras.Model(inputs=[image_input, sensor_input], outputs=[steering_output, throttle_output])
    return model

# -----------------------------------------------------------------------------
# 步驟 2: 模擬你的數據
#
# 由於我無法存取你實際的檔案，這裡生成一些模擬數據來演示。
# 在實際使用時，請替換成你自己的 `X_train_img` 和 `X_train_sensor`。
# -----------------------------------------------------------------------------
def generate_dummy_data(num_samples=100, img_shape=(356, 634, 3), sensor_dim=12):
    """生成模擬的圖像和感測器數據"""
    # 這裡的模擬數據與你的程式碼相符，是正規化過的 (0-1之間)
    dummy_images = np.random.rand(num_samples, *img_shape).astype(np.float32)
    dummy_sensors = np.random.rand(num_samples, sensor_dim).astype(np.float32)
    return dummy_images, dummy_sensors

# 定義你的 12 個感測器名稱
# 這是從你的 parse_sensor_data_from_log 函式中推斷出來的順序
SENSOR_FEATURE_NAMES = [
    'Velocity', 'Rot_X', 'Rot_Y', 'Rot_Z',
    'AngularVelocity_X', 'AngularVelocity_Y', 'AngularVelocity_Z',
    'rollAngle', 'DistanceFront', 'DistanceRear', 'DistanceLeft',
    'DistanceRight'
]

def main():
    print("✅ 正在創建並載入模型...")
    model = create_autodrive_model()
    # 這裡假設你的模型權重已載入，但在範例中我們將使用未訓練的模型
    # model.load_weights('./checkpoints/model_epoch_008.h5')

    # 生成訓練數據和測試數據
    # SHAP 需要一個背景數據 (background data) 來估計基準值
    X_train_img, X_train_sensor = generate_dummy_data(num_samples=500)
    X_test_img, X_test_sensor = generate_dummy_data(num_samples=50)
    
    # 準備 SHAP 所需的背景數據
    # DeepExplainer 對於大規模的背景數據會非常慢，通常取一小部分即可
    background_images = X_train_img[np.random.choice(X_train_img.shape[0], 50, replace=False)]
    background_sensors = X_train_sensor[np.random.choice(X_train_sensor.shape[0], 50, replace=False)]
    
    # -----------------------------------------------------------------------------
    # 步驟 3: 初始化 SHAP 解釋器
    #
    # 這是核心步驟！對於多輸入模型，background 參數需要是一個列表。
    # -----------------------------------------------------------------------------
    print("🔬 正在初始化 SHAP 解釋器... (這可能需要一些時間)")
    # `model.inputs` 是一個列表，所以 `background` 也必須是列表
    explainer = shap.DeepExplainer(
        model, 
        [background_images, background_sensors]
    )

    # -----------------------------------------------------------------------------
    # 步驟 4: 計算 SHAP 值
    #
    # 計算測試數據的 SHAP 值。
    # -----------------------------------------------------------------------------
    print("🔍 正在計算測試數據的 SHAP 值...")
    # shap_values 是一個列表，因為模型有兩個輸出 (steering, throttle)
    shap_values = explainer.shap_values(
        [X_test_img, X_test_sensor]
    )
    
    # -----------------------------------------------------------------------------
    # 步驟 5: 視覺化解釋結果
    #
    # 這裡分別為「轉向角」和「油門」輸出生成解釋圖表。
    # -----------------------------------------------------------------------------
    print("\n📈 正在生成 SHAP 視覺化圖表...")

    # --- 轉向角 (Steering) 視覺化 ---
    print("  -> 轉向角 (Steering) SHAP 分析")
    # `shap_values[0]` 對應第一個輸出 (轉向角)
    # `shap_values[0]` 也是一個列表，包含圖像和感測器的 SHAP 值
    steering_shap_image = shap_values[0][0]
    steering_shap_sensor = shap_values[0][1]

    # 全局解釋: 總結所有感測器對轉向角預測的整體影響
    # 這就是你想要的「整體」解釋！
    print("    -> 顯示所有感測器的整體重要性 (Summary Plot)")
    shap.summary_plot(
        steering_shap_sensor, 
        X_test_sensor, 
        feature_names=SENSOR_FEATURE_NAMES,
        show=False
    )
    plt.title("Steering Prediction: Sensor Feature Importance (SHAP Summary)")
    plt.tight_layout()
    plt.savefig("steering_shap_summary.png")
    plt.show()

    # 局部解釋: 顯示單一預測的 SHAP 值
    # 這裡選擇第一個測試樣本來解釋
    print("    -> 顯示單一預測的感測器貢獻 (Force Plot)")
    shap.initjs() # 啟動 JavaScript 視覺化
    force_plot_html = shap.force_plot(
        explainer.expected_value[0], # 基準值 (expected_value)
        steering_shap_sensor[0],     # 選擇第一個樣本的 SHAP 值
        X_test_sensor[0],            # 選擇第一個樣本的感測器數值
        feature_names=SENSOR_FEATURE_NAMES,
        show=False,
        matplotlib=True
    )
    
    plt.title("Steering Prediction: Single Instance Explanation (SHAP Force Plot)")
    plt.tight_layout()
    plt.savefig("steering_shap_force_plot.png")
    plt.show()

    # --- 油門 (Throttle) 視覺化 ---
    print("\n  -> 油門 (Throttle) SHAP 分析")
    throttle_shap_sensor = shap_values[1][1]

    print("    -> 顯示所有感測器的整體重要性 (Summary Plot)")
    shap.summary_plot(
        throttle_shap_sensor, 
        X_test_sensor, 
        feature_names=SENSOR_FEATURE_NAMES,
        show=False
    )
    plt.title("Throttle Prediction: Sensor Feature Importance (SHAP Summary)")
    plt.tight_layout()
    plt.savefig("throttle_shap_summary.png")
    plt.show()

    print("✅ SHAP 分析腳本已執行完成。")
    print("✅ 圖像已儲存為 steering_shap_summary.png, steering_shap_force_plot.png 和 throttle_shap_summary.png")

if __name__ == '__main__':
    main()
