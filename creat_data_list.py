import os
import random

def parse_log_file(log_path):
    file_list = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.startswith("StartDateTime") or line.startswith("Timestamp") or line.strip() == "":
                continue
            parts = line.strip().split(';')
            if len(parts) >= 3:
                timestamp = parts[0]
                values = parts[2].split(',')
                if len(values) >= 1:
                    angle = values[-1]  # 取最後一個值（steering angle）
                    throttle = values[-2]  # throttle
                    file_list.append((timestamp, angle,throttle))
    return file_list

def match_images(image_folder, file_list):
    matched_list = []
    for timestamp, angle,throttle in file_list:
        # 將 timestamp 轉為圖檔名稱
        filename = f"Screenshot_{timestamp}.jpg"
        img_path = os.path.join(image_folder, filename)
        if os.path.exists(img_path):
            matched_list.append((img_path.replace('\\', '/'), angle,throttle))
        else:
            print(f"[⚠️] 圖片不存在: {img_path}")
    return matched_list

def write_txt(file_list, save_dir, mode='train'):
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, f"{mode}.txt")
    with open(txt_path, 'w') as f:
        for img_path, angle,throttle in file_list:
            f.write(f"{img_path};{angle};{throttle}\n")
    print(f"✅ 已生成 {mode}.txt，共 {len(file_list)} 筆")

def main():
    log_path = 'C:/Users/Clary Lin/Downloads/20250706Car01/20250506Car01/Car0706/Event/Event_20250706_221950.txt'   # log檔路徑
    image_folder = 'C:/Users/Clary Lin/Downloads/20250706Car01/20250506Car01/Car0706/Cam01/20250706_221950'         # 照片資料夾
    save_dir = './data'                    # 輸出 train.txt/val.txt 的位置
    train_ratio = 0.8

    raw_list = parse_log_file(log_path)
    matched = match_images(image_folder, raw_list)

    random.seed(42)
    random.shuffle(matched)
    split_idx = int(len(matched) * train_ratio)
    train_list = matched[:split_idx]
    val_list = matched[split_idx:]

    write_txt(train_list, save_dir, mode='train')
    write_txt(val_list, save_dir, mode='val')

if __name__ == "__main__":
    main()
