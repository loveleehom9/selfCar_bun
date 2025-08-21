import os
import random
import config
# add by bun 測試分層抽樣
from sklearn.model_selection import train_test_split # 新增: 導入 train_test_split
import numpy as np

def parse_log_file(log_path):
    file_list = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.startswith("StartDateTime") or line.startswith("Timestamp") or line.strip() == "":
                continue
            parts = line.strip().split(';')
            if len(parts) >= 3:
                timestamp = f"{float(parts[0]):.2f}"
                values = parts[2].split(',')
    #             if len(values) >= 1:
    #                 angle = values[-1]  # 取最後一個值（steering angle）
    #                 throttle = values[-2]  # throttle
    #                 file_list.append((timestamp, angle,throttle))
    # return file_list
                if len(values) >= 2:
                    throttle = values[1].strip()
                    angle = values[2].strip()
                    file_list.append((timestamp, angle, throttle))
                else:
                    print(f"⚠️ 值長度不足 at {timestamp}: {parts[2]}")
            else:
                print(f"⚠️ 欄位不足: {line.strip()}")
    return file_list

def parse_sensor_file(sensor_path):
    sensor_dict = {}
    with open(sensor_path, 'r') as f:
        for line in f:
            if line.startswith("StartDateTime") or line.startswith("Timestamp") or line.strip() == "":
                continue

            parts = line.strip().split(';')

            try:
                timestamp = f"{float(parts[0].strip()):.2f}"
            except ValueError:
                continue

            # 初始化欄位
            Velocity = ""
            rot = ""
            ang_vel = ""
            rollAngle = ""
            DistanceFront = ""
            DistanceRear = ""
            DistanceLeft = ""
            DistanceRight = ""


            for part in parts:
                part = part.strip()
                if part.startswith("Velocity:"):
                    Velocity = part.replace("Velocity:", "").strip()
                elif part.startswith("Rot:"):
                    rot = part.replace("Rot:", "").strip()
                elif part.startswith("AngularVelocity:"):
                    ang_vel = part.replace("AngularVelocity:", "").strip()
                elif part.startswith("rollAngle:"):
                    rollAngle = part.replace("rollAngle:", "").strip()
                elif part.startswith("DistanceFront:"):
                    DistanceFront = part.replace("DistanceFront:", "").strip()
                elif part.startswith("DistanceRear:"):
                    DistanceRear = part.replace("DistanceRear:", "").strip()
                elif part.startswith("DistanceLeft:"):
                    DistanceLeft = part.replace("DistanceLeft:", "").strip()
                elif part.startswith("DistanceRight:"):
                    DistanceRight = part.replace("DistanceRight:", "").strip()


            if Velocity and rot and ang_vel and rollAngle and DistanceFront and DistanceRear and DistanceLeft and DistanceRight:
                sensor_dict[timestamp] = (Velocity, rot, ang_vel, rollAngle, DistanceFront, DistanceRear, DistanceLeft,DistanceRight)
            else:
                print(f"⚠️ 缺資料 at {timestamp} — Velocity: {Velocity}, rot: {rot}, ang_vel: {ang_vel}, rollAngle: {rollAngle}, DistanceFront: {DistanceFront}, DistanceRear: {DistanceRear}, DistanceLeft: {DistanceLeft}, DistanceRight: {DistanceRight}")

    print(f"\n✅ sensor_dict 成功擷取 {len(sensor_dict)} 筆")
    return sensor_dict





def match_images(image_folder, file_list, sensor_data):
    matched_list = []
    for timestamp, angle, throttle in file_list:
        # 修正精度，四捨五入至小數點第 2 位
        rounded_timestamp = f"{float(timestamp):.2f}"
        filename = f"Screenshot_{rounded_timestamp}.jpg"
        img_path = os.path.join(image_folder, filename)

        if os.path.exists(img_path) and rounded_timestamp in sensor_data:
            Velocity ,rot ,ang_vel ,rollAngle ,DistanceFront ,DistanceRear ,DistanceLeft ,DistanceRight = sensor_data[rounded_timestamp]
            matched_list.append((img_path.replace('\\', '/'), angle, throttle,  Velocity ,rot ,ang_vel ,rollAngle ,DistanceFront ,DistanceRear ,DistanceLeft ,DistanceRight))
        else:
            print(f"[⚠️] 缺少圖片或 sensor 資料: {filename}")
    return matched_list

def write_txt(file_list, save_dir, mode='train'):
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, f"{mode}.txt")
    with open(txt_path, 'w') as f:
        for img_path, angle,throttle,Velocity ,rot ,ang_vel ,rollAngle ,DistanceFront ,DistanceRear ,DistanceLeft ,DistanceRight in file_list:
            f.write(f"{img_path};{angle};{throttle};{Velocity};{rot};{ang_vel};{rollAngle};{DistanceFront};{DistanceRear};{DistanceLeft};{DistanceRight}\n")
    print(f"✅ 已生成 {mode}.txt，共 {len(file_list)} 筆")

def main():
    log_path = config.LOG_PATH # 'C:/Users/User/source/repos/Car02/Event/Event_20250725_110504.txt'   # log檔路徑
    sensor_path = config.SENSOR_PATH # 'C:/Users/User/source/repos/Car02/Log/Log_20250725_110504.txt'   # sensor檔路徑
    image_folder = config.IMAGE_FOLDER # 'C:/Users/User/source/repos/Car02/Cam01/20250725_110504'         # 照片資料夾
    save_dir = config.SAVE_DIR # './data'     # 輸出 train.txt/val.txt 的位置
    train_ratio = config.TRAIN_RATIO

    # 載入所有原始數據
    raw_list = parse_log_file(log_path)
    sensor_data = parse_sensor_file(sensor_path)
    matched = match_images(image_folder, raw_list, sensor_data)
    print(f"✅ 成功匹配所有原始數據，共 {len(matched)} 筆")

    print("\n🔍 file_list 前 5 筆：")
    for i, (ts, angle, throttle) in enumerate(raw_list[:5]):
        print(f"{ts} → angle: {angle}, throttle: {throttle}")

    print("\n🔍 sensor_data 前 5 筆：")
    for i, (ts, values) in enumerate(sensor_data.items()):
        print(f"{ts} → {values}")
        if i >= 4:
            break

    print("\n📁 圖片檔名預覽前 5：")
    for f in os.listdir(image_folder)[:5]:
        print(f)

    # Mark by bun to change split
    """
    random.seed(42)
    random.shuffle(matched)
    """
    

    # 利用轉向角作為分層抽樣
    # 取得轉向角並進行分桶
    angles = np.array([float(data[1]) for data in matched])
    
    # 將轉向角做 等寬距離 的取樣
    min_angle = min(angles)
    max_angle = max(angles)
    num_bins = 60 # 60 30 20 10 都可能造成某個區間樣本數過少，可能是資料集的關係，擴大區間。
    
    # 初始化用於分層和直接加入訓練集的列表
    data_to_stratify = []
    stratify_bins = []
    data_to_add_to_train = []

    # 如果轉向角範圍為 0 (所有角度都一樣)，則無需分層
    if min_angle == max_angle:
        print("💡 提示: 所有轉向角都相同，無需分層抽樣。")
        binned_angles = None
        data_to_stratify = matched
    else:
        bins = np.linspace(min_angle, max_angle, num_bins + 1)
        binned_angles_all = np.digitize(angles, bins)
        
        # 統計每個分桶的樣本數
        unique_bins, counts = np.unique(binned_angles_all, return_counts=True)
        bin_counts = dict(zip(unique_bins, counts))
        
        # 根據分桶樣本數將數據分類
        for i, data_point in enumerate(matched):
            bin_num = binned_angles_all[i]
            if bin_counts[bin_num] >= 2:
                data_to_stratify.append(data_point)
                stratify_bins.append(bin_num)
            else:
                data_to_add_to_train.append(data_point)

        if data_to_add_to_train:
            print(f"⚠️ 警告: 發現 {len(data_to_add_to_train)} 個樣本數不足的分桶數據，它們將被直接加入訓練集。")
            
        print(f"✅ 將 {len(data_to_stratify)} 筆數據用於分層劃分。")
        
    # 3. 執行訓練集和驗證集劃分 (使用分層抽樣，如果可行)
    # 現在 data_to_stratify 和 stratify_bins 的長度保證一致
    if data_to_stratify:
        train_list_stratified, val_list = train_test_split(
            data_to_stratify,
            test_size=1 - train_ratio,
            random_state=42,
            stratify=stratify_bins
        )
    else:
        # 如果沒有可分層的數據，所有數據都直接進訓練集
        train_list_stratified = data_to_add_to_train
        val_list = []
    
    # 將單獨保存的數據點合併到訓練集中
    train_list = train_list_stratified + data_to_add_to_train
    if train_list:
        random.shuffle(train_list) # 隨機打亂，以避免這些數據點集中在訓練集末尾

    print(f"✅ 數據劃分完成：訓練集 {len(train_list)} 筆，驗證集 {len(val_list)} 筆")

    # Mark bu bun to change split
    """
    split_idx = int(len(matched) * train_ratio)
    train_list = matched[:split_idx]
    val_list = matched[split_idx:]
    """
    # 將劃分後的數據寫入 TXT 文件
    write_txt(train_list, save_dir, mode='train')
    write_txt(val_list, save_dir, mode='val')

if __name__ == "__main__":
    main()
