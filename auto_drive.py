import socket
import time
import tensorflow as tf
import numpy as np
import cv2
import json

# 初始化模型
model = tf.keras.models.load_model(".\checkpoints\model_epoch_008.h5")

# 設定 TCP socket
sock_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_receive.bind(("192.168.0.12", 12351))  # 接收所有來源的訊息

print("✅ 等待來自 Unity 的影像...")

# 設定 TCP socket 用來發送控制數據回 Unity
sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

while True:
    try:
        # Step 1️⃣: 先接收控制訊息（目前沒用，可略過）
        control_data, addr = sock_receive.recvfrom(1024)
        print("🛰️ 收到控制訊息:", control_data.decode())  # 可略過但會同步

        # Step 2️⃣: 再接收圖片（JPEG壓縮）
        image_data, _ = sock_receive.recvfrom(65536)
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        print("🛰️ 收到控制圖片")  


        if img is None:
            print("❌ 無法解碼圖像")
            continue

        # 預處理（依模型輸入）
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 預測 steering
        prediction = model.predict(img, batch_size=1)[0]
        steering = float(prediction[0])
        throttle = float(prediction[1])

        steering = float(np.clip(steering, -1, 1))
        throttle = float(np.clip(throttle, 0, 1))

        # 打包成 JSON 傳回 Unity
        control_dict = {"steering": steering, "throttle": throttle}
        control_json = json.dumps(control_dict)
        # 設定 Unity 端的地址（需要替換為 Unity 端的 IP 和端口）
        sock_send.connect(("192.168.0.12", 12352))  # 這是 Unity 端的接收端口
        sock_send.sendall(control_json.encode())

        print(f"📤 傳回控制：{control_json}")
        time.sleep(0.05)  # 控制傳輸頻率


    except Exception as e:
        print("⚠️ 錯誤:", e)
