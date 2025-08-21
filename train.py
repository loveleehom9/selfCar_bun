import os
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import Huber
# Add by bun try EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard , EarlyStopping , ReduceLROnPlateau
from models import create_autodrive_model
from datasets import load_data  # 你之前寫的 tf.data.Dataset 封裝
import config

# add by bun
# 設定寬度與高度及通道資訊
input_height = config.TARGET_IMAGE_HEIGHT
input_width = config.TARGET_IMAGE_WIDTH
input_channels = config.INPUT_CHANNELS
batch_Size = config.BATCH_SIZE
Epochs = config.NUM_EPOCHS
learning_Rate = config.LEARNING_RATE
sensor_input_dim = config.SENSOR_INPUT_DIM 

### Add by bun 114/07/25 目前測試僅有 tensorflow 2.10.1 版，可以正常運作GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 設置 GPU 記憶體採動態分配，這樣不會一次性預留全部記憶體空間
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        print(f"TensorFlow will use GPU: {gpus[0].name}") # 確認有使用的GPU
    except RuntimeError as e:
        # 列印錯誤資訊
        print(e)
else:
    print("No GPU detected by TensorFlow. Running on CPU.")
### Add End by bun 114/07/25

def train(data_folder='./data',
          log_dir='logs',
          ckpt_dir='checkpoints',
          batch_size=batch_Size ,
          epochs=Epochs ,
          lr=learning_Rate,
          resume_from=None):

    # 準備目錄
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 載入資料
    # modify by bun load_data(path,augment=True)，設定True時，會開啟數據增強功能
    # load_data 函數需要返回圖像數據、感測器數據以及轉向和油門目標值。
    # load_data 現在返回的資料是 (image, sensor_data, steering_target, throttle_target)
    print("🚀 載入訓練數據集...") 
    # buffer_size=1024
    train_dataset = load_data(os.path.join(data_folder, 'train.txt'),True).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("🚀 載入驗證數據集...")
    val_dataset = load_data(os.path.join(data_folder, 'val.txt'),False).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print("\n✅ 數據集載入成功。預覽數據形狀：")
    for inputs, outputs in train_dataset.take(1):
        print("  - 訓練集輸入形狀: ", {key: value.shape for key, value in inputs.items()})
        print("  - 訓練集輸出形狀: ", {key: value.shape for key, value in outputs.items()})
        break

    # 建立或載入模型
    if resume_from is not None:
        model = tf.keras.models.load_model(resume_from)
        print(f"✅ 已載入預訓練模型：{resume_from}")
    else:
        #model = create_autodrive_model(input_shape=(input_height, input_width, input_channels))
        model = create_autodrive_model(input_shape=(input_height, input_width, input_channels), sensor_input_dim=sensor_input_dim)

    model.summary()

    # 編譯模型
    # Modify by bun to change output 
    # 採用兩個輸出，需要為每個輸出指定損失函數。
    # Key name 必須與 models.py 中定義的輸出層的 name 屬性完全一致。
    # model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())
    # Add by bun want try 
    # model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1.0)) # 初步測試 使用 delta 1.0，可再測試0.5或0.3
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={
            'steering_output': Huber(delta=1.0, name='huber_steering_loss'), # 轉向的損失
            'throttle_output': Huber(delta=1.0, name='huber_throttle_loss')  # 油門的損失
        },
        # 可以為不同的損失設定個別權重，例如給轉向角更多的權重
        # loss_weights 可以調整不同損失函數的重要性，這裡先保持相等，越小越重要
        loss_weights={
            'steering_output': 0.5,
            'throttle_output': 1.0,
        },
        # 為每個輸出添加監控指標
        metrics={
            'steering_output': ['mae', 'mse'], # 轉向可以監控平均絕對誤差和均方誤差
            'throttle_output': ['mae', 'mse']   # 油門也是監控這些指標
        }
    )

    # Callbacks：TensorBoard + Checkpoint
    log_path = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        TensorBoard(log_dir=log_path),
        # 監控 'val_loss' 是監控所有輸出的總驗證損失
        ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, 'model_epoch_{epoch:03d}.h5'),
            save_best_only=False, # 多輸出模型通常不建議只保存 'best'，因為 'best' 的定義會變得複雜
            save_weights_only=False,
            # 對於多輸出模型，moniter 通常需要指定某個輸出的損失，例如 'val_steering_output_loss'
            # 或者可以監控總損失 'val_loss'
            monitor='val_loss',
            verbose=1
        )

        # Add by bun to Early Stopping
        ,EarlyStopping(
            monitor='val_loss',
            patience=5,  # 之後可以從 config 中讀取 從10改成 5 ，曾快速度
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, # 當驗證損失沒有改善時，將學習率乘以 0.5
            patience=5,  # 連續 5 個 epoch 驗證損失沒有改善後觸發
            min_lr=1e-6, # 最小學習率限制
            verbose=1
        )
    ]

    print("\n🏁 開始訓練模型...")
    # 訓練
    # 模型調整成有兩個輸入和兩個輸出，
    # 因此 train_dataset 和 val_dataset 的元素結構需要是 ((image, sensor_data), (steering_target, throttle_target)) 
    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=epochs,
              callbacks=callbacks)

    print("✅ 訓練完成！")


if __name__ == '__main__':
    train()
