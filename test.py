# import os

# # 指定文件夹路径
# folder_path = "/home/Users/xsc/experiment/SCL/FF++/DF/unlabled_data"

# # 获取文件夹下的所有文件
# image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# # 处理每个图像文件
# for image_file in image_files:
#     # 构建完整的文件路径
#     file_path = os.path.join(folder_path, image_file)
    
#     # 使用下划线 "_" 分割字符串
#     parts = image_file.split("_")
    
#     # 最后一个部分保留，前面的部分合并
#     new_name = f"{parts[0]}{ ''.join(parts[1:-1])}_{parts[-1]}"
    
#     # 构建新的文件路径
#     new_path = os.path.join(folder_path, new_name)
    
#     # 重命名文件
#     os.rename(file_path, new_path)

# print("文件名已更改完成。")

# import os
# import random
# import shutil

# def copy_random_files(source_folder, destination_folder, num_files):
#     # 获取源文件夹下的所有文件
#     all_files = os.listdir(source_folder)
    
#     # 从所有文件中随机选择指定数量的文件
#     selected_files = random.sample(all_files, num_files)
    
#     # 确保目标文件夹存在，如果不存在则创建
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
    
#     # 复制选定的文件到目标文件夹
#     for file_name in selected_files:
#         source_path = os.path.join(source_folder, file_name)
#         destination_path = os.path.join(destination_folder, file_name)
#         shutil.copyfile(source_path, destination_path)

# # 指定源文件夹和目标文件夹
# source_folder = "/home/Users/xsc/experiment/SCL/FF++/DF/unlabled_data"
# destination_folder = "/home/Users/xsc/experiment/SCL/FF++/DF/unlabled_data_less"

# # 指定要复制的文件数量
# num_files_to_copy = 1000

# # 调用函数进行文件复制
# copy_random_files(source_folder, destination_folder, num_files_to_copy)

import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_labels(image_name):
    actual_label = image_name[0]  # Extract the first character as the actual label
    return actual_label

def calculate_accuracy(predictions):
    correct_predictions = sum(predictions)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def main():
    file_path = '/home/Users/xsc/experiment/Unsupervised_DF_Detection-main/sava_result/pseudo_label.json'  # Replace with the actual path to your JSON file
    data = load_json(file_path)

    total_units = len(data)
    actual_labels = []
    predicted_labels = []

    for entry in data:
        image_name = entry["image name"]
        actual_label = extract_labels(image_name)
        predicted_label = entry["image label"]

        actual_labels.append(actual_label)
        predicted_labels.append(int(actual_label == predicted_label))  # 1 if correct prediction, 0 otherwise

    accuracy = calculate_accuracy(predicted_labels)

    print(f'Total units: {total_units}')
    print(f'Actual labels: {actual_labels}')
    print(f'Predicted labels: {predicted_labels}')
    print(f'Accuracy (ACC): {accuracy:.2%}')

if __name__ == "__main__":
    main()
