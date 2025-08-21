from PIL import Image
import os

# 替換成你自己的資料夾路徑
folder_path = 'C:/Users/Clary Lin/Downloads/20250706Car01/20250506Car01/Car0706/Cam01/20250706_221950'

# # 建立輸出資料夾（如果需要的話）
# output_folder = os.path.join(folder_path, "converted_jpg")
# os.makedirs(output_folder, exist_ok=True)

# 遍歷所有 PNG 檔案
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.png'):
        png_path = os.path.join(folder_path, filename)
        
        # 開啟圖片並轉成 RGB（避免透明背景錯誤）
        with Image.open(png_path) as img:
            rgb_img = img.convert('RGB')
            
            # 設定輸出檔案名稱（同名但 .jpg 副檔名）
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(folder_path, jpg_filename)
            
            # 儲存為 JPG
            rgb_img.save(jpg_path, 'JPEG')
            print(f"✅ Converted: {filename} → {jpg_filename}")
