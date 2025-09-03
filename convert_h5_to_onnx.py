import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
import numpy as np
import os
import config

# === 1. 基本參數設定 ===
H5_PATH = config.MODEL_PATH        # 輸入的 Keras .h5 檔案名稱
ONNX_PATH = "./car_control_model.onnx"    # 輸出的 ONNX 檔案名稱

# === 2. 載入 Keras 模型 ===
print(f"[INFO] Loading Keras model from {H5_PATH} ...")
model = tf.keras.models.load_model(H5_PATH)
print("[INFO] Model loaded successfully.")

# === 3. 轉換為 SavedModel (tf2onnx 轉換器需用) ===
SAVED_MODEL_DIR = "saved_model_temp"
print(f"[INFO] Saving model as TensorFlow SavedModel to {SAVED_MODEL_DIR} ...")
model.save(SAVED_MODEL_DIR)
print("[INFO] SavedModel exported.")

# === 4. 轉換為 ONNX 格式 ===
print(f"[INFO] Converting SavedModel to ONNX: {ONNX_PATH} ...")
spec = (tf.TensorSpec((None, 356, 634, 3), tf.float32, name="image_input"),  # 輸入1：影像
        tf.TensorSpec((None, 12), tf.float32, name="sensor_input"))           # 輸入2：感測器
output_path = ONNX_PATH

# 執行轉換
"""
model_proto, _ = tf2onnx.convert.from_saved_model(
    SAVED_MODEL_DIR, 
    input_signature=spec, 
    output_path=output_path,
    opset=15
)
"""
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=15, output_path=output_path)

print(f"[SUCCESS] Model converted and saved to {ONNX_PATH}")

# === 5. 驗證 ONNX 檔案可用性 ===
print("[INFO] Verifying the ONNX model ...")
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("[INFO] ONNX model structure is valid.")

# === 6. 使用 ONNX Runtime 測試推論 ===

# 製作假資料（與模型輸入一致）
dummy_image = np.random.rand(1, 356, 634, 3).astype(np.float32)   # 測試影像
dummy_sensor = np.random.rand(1, 12).astype(np.float32)           # 測試感測器

# 取得 ONNX 輸入名稱
ort_session = ort.InferenceSession(ONNX_PATH)
input_names = [inp.name for inp in ort_session.get_inputs()]
print(f"[INFO] ONNX model input names: {input_names}")

# 組裝推論輸入
onnx_inputs = {
    input_names[0]: dummy_image,
    input_names[1]: dummy_sensor,
}
# 執行推論
print("[INFO] Running ONNX inference test ...")
outputs = ort_session.run(None, onnx_inputs)
print(f"[SUCCESS] ONNX model inference output: {outputs}")

# === 7. 清理暫存資料夾 ===
import shutil
shutil.rmtree(SAVED_MODEL_DIR)
print("[INFO] Temporary SavedModel directory cleaned up.")

print("\n=== All steps completed successfully! ===")