# 以下是一个简单的Python脚本示例，它遍历指定文件夹中的所有图像文件，然后去除文件名中的前缀，并保持文件的其余部分和格式不变

import os

# 指定你的数据集文件夹路径
dataset_folder_path = r'/home/wen/data_3/XianYu/2024_08/OURNet-main/datasets/LOLv1v2/LOLv2/Synthetic/Train/glow_images'

# 指定要去除的前缀
prefix_to_remove = 'glow_'

# 遍历文件夹中的所有文件
for filename in os.listdir(dataset_folder_path):
    if filename.startswith(prefix_to_remove):
        # 构造新的文件名，去除前缀
        new_filename = filename[len(prefix_to_remove):]
        # 获取文件的完整原路径
        original_path = os.path.join(dataset_folder_path, filename)
        # 获取文件的新路径
        new_path = os.path.join(dataset_folder_path, new_filename)
        # 重命名文件
        os.rename(original_path, new_path)
        print(f'Renamed {filename} to {new_filename}')






















