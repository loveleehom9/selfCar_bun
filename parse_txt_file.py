import matplotlib.pyplot as plt
import numpy as np
import os
import config

from datasets import parse_txt_file

train_dir_path = config.TRAIN_DIR_PATH # './data/train.txt'

data_entries = parse_txt_file(train_dir_path)

# 讀取所有資訊
img_paths = [entry[0] for entry in data_entries]
angles = [entry[1] for entry in data_entries]
throttles = [entry[2] for entry in data_entries]
velocities = [entry[3] for entry in data_entries]

# 感測器數據
# rotation (x, y, z)
rot_x = [entry[4][0] for entry in data_entries]
rot_y = [entry[4][1] for entry in data_entries]
rot_z = [entry[4][2] for entry in data_entries]

# angular_velocity (x, y, z)
ang_vel_x = [entry[5][0] for entry in data_entries]
ang_vel_y = [entry[5][1] for entry in data_entries]
ang_vel_z = [entry[5][2] for entry in data_entries]

roll_angles = [entry[6] for entry in data_entries]
dist_front = [entry[7] for entry in data_entries]
dist_rear = [entry[8] for entry in data_entries]
dist_left = [entry[9] for entry in data_entries]
dist_right = [entry[10] for entry in data_entries]

bins_All = 200

# 繪製角度直方圖
plt.figure(figsize=(10, 5))
plt.hist(angles, bins=bins_All, color='skyblue', edgecolor='black')
plt.title('Distribution of Steering Angles in Training Data')
plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 繪製油門直方圖
plt.figure(figsize=(10, 5))
plt.hist(throttles, bins=bins_All, color='lightcoral', edgecolor='black')
plt.title('Distribution of Throttle in Training Data')
plt.xlabel('Throttle')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 繪製速度直方圖
velocities = [entry[3] for entry in data_entries]
plt.figure(figsize=(10, 5))
plt.hist(velocities, bins=bins_All, color='lightgreen', edgecolor='black')
plt.title('Distribution of Velocity in Training Data')
plt.xlabel('Velocity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# 繪製直方圖
def plot_histogram(data, title, xlabel, color='skyblue', bins=bins_All):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# --- 繪製所有感測器數據的直方圖 ---
print(" 正在繪製感測器數據的直方圖...")

plot_histogram(roll_angles, 'Distribution of Roll Angle in Training Data', 'Roll Angle', color='gold')
plot_histogram(dist_front, 'Distribution of Front Distance in Training Data', 'Distance Front', color='purple')
plot_histogram(dist_rear, 'Distribution of Rear Distance in Training Data', 'Distance Rear', color='orange')
plot_histogram(dist_left, 'Distribution of Left Distance in Training Data', 'Distance Left', color='teal')
plot_histogram(dist_right, 'Distribution of Right Distance in Training Data', 'Distance Right', color='maroon')

# 旋轉數據 (Rotation - Euler Angles)
plot_histogram(rot_x, 'Distribution of Rotation X (Pitch) in Training Data', 'Rotation X (Pitch)', color='blue')
plot_histogram(rot_y, 'Distribution of Rotation Y (Yaw) in Training Data', 'Rotation Y (Yaw)', color='green')
plot_histogram(rot_z, 'Distribution of Rotation Z (Roll) in Training Data', 'Rotation Z (Roll)', color='red')

# 角速度數據 (Angular Velocity)
plot_histogram(ang_vel_x, 'Distribution of Angular Velocity X in Training Data', 'Angular Velocity X', color='cyan')
plot_histogram(ang_vel_y, 'Distribution of Angular Velocity Y in Training Data', 'Angular Velocity Y', color='magenta')
plot_histogram(ang_vel_z, 'Distribution of Angular Velocity Z in Training Data', 'Angular Velocity Z', color='lime')

print(" 所有感測器數據直方圖已生成。")

# 繪製散點圖，可再自行新增
# distFront vs throttle
plt.figure(figsize=(10, 8))
plt.scatter(dist_front, throttles, alpha=0.5, s=5)
plt.title('Throttle vs. Front Distance')
plt.xlabel('Distance Front')
plt.ylabel('Throttle')
plt.grid(True)
plt.show()
