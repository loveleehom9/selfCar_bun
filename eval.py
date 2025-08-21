import os
import time
import numpy as np
import tensorflow as tf
import cv2
from datasets import load_data, parse_txt_file
from models import create_autodrive_model

def evaluate(model_path, txt_path='data/val.txt', top_k=10, save_dir='eval_results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # 設備
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"✅ 使用設備：{device}")

    # 載入原始檔案清單，為了取得圖片路徑
    raw_data = parse_txt_file(txt_path)

    # 載入模型與資料 
    # modify by bun
    val_dataset = load_data(txt_path,False).batch(1)
    model = tf.keras.models.load_model(model_path)
    print(f"✅ 已載入模型：{model_path}")

    loss_fn = tf.keras.losses.MeanSquaredError()
    total_loss = 0.0
    total_count = 0
    start_time = time.time()

    # 儲存誤差資訊
    errors = []

    for i, (batch, raw) in enumerate(zip(val_dataset, raw_data)):
        inputs, y_true = batch
        preds = model(inputs, training=False)
        loss = loss_fn(y_true, preds).numpy()
        total_loss += loss
        total_count += 1

        # 印出預測 vs 真實值
        print(f"[{i}] 🎯 GT: {y_true.numpy()[0]} | 🔮 Pred: {preds.numpy()[0]} | ❌ Loss: {loss:.4f}")

        # 存下誤差、圖片路徑、預測值
        errors.append({
            'idx': i,
            'img_path': raw[0],  # image path
            'gt': y_true.numpy()[0],
            'pred': preds.numpy()[0],
            'loss': loss
        })

    # 統計
    avg_loss = total_loss / total_count
    avg_time = (time.time() - start_time) / total_count
    print(f"\n📉 驗證集 MSE：{avg_loss:.4f}")
    print(f"⏱️ 平均推論時間：{avg_time:.4f} 秒")

    # 儲存 top K 錯誤圖片
    errors.sort(key=lambda x: x['loss'], reverse=True)
    top_errors = errors[:top_k]

    for idx, item in enumerate(top_errors):
        img = cv2.imread(item['img_path'])
        if img is None:
            continue
        h, w = img.shape[:2]
        info = f"GT: {item['gt']}, Pred: {item['pred']}, Loss: {item['loss']:.4f}"
        cv2.putText(img, info, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        save_path = os.path.join(save_dir, f"error_{idx:02d}.jpg")
        cv2.imwrite(save_path, img)

    print(f"📂 Top {top_k} 錯誤圖片已儲存至：{save_dir}")

if __name__ == '__main__':
    evaluate(model_path='./checkpoints/model_epoch_008.h5', txt_path='./data/val.txt', top_k=10)
