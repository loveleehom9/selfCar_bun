# import os
# import tensorflow as tf
# import numpy as np
# from datasets import load_data  # 你之前寫的 tf.data.Dataset 封裝
# from tensorflow.keras.losses import MeanSquaredError

# # 評估一個模型，回傳 MSE
# def evaluate_model(model_path, val_dataset):
#     model = tf.keras.models.load_model(model_path)
#     mse_loss = MeanSquaredError()

#     total_loss = 0.0
#     count = 0

#     for batch in val_dataset:
#         images, sensor_data = batch
#         preds = model(images, training=False)
#         loss = mse_loss(sensor_data, preds).numpy()
#         total_loss += loss * images.shape[0]
#         count += images.shape[0]

#     avg_mse = total_loss / count
#     return avg_mse

# # 主流程
# def evaluate_all_models(model_dir='./checkpoints', val_txt='./data/val.txt', batch_size=64):
#     # 準備驗證資料
#     val_dataset = load_data(val_txt).batch(batch_size).prefetch(tf.data.AUTOTUNE)

#     # 找出所有模型
#     model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.h5')])
#     best_mse = float('inf')
#     best_model = None

#     print("🔍 開始驗證所有模型...\n")
#     for model_file in model_files:
#         model_path = os.path.join(model_dir, model_file)
#         mse = evaluate_model(model_path, val_dataset)
#         print(f"📊 {model_file} → MSE: {mse:.4f}")

#         if mse < best_mse:
#             best_mse = mse
#             best_model = model_file

#     print("\n✅ 最佳模型：", best_model)
#     print(f"🎯 最低 MSE：{best_mse:.4f}")

# if __name__ == '__main__':
#     evaluate_all_models()
