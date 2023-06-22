import os

# 定义文件夹路径
train_folder = "train"
train_labels_folder = "train_labels"

# 创建新的文件夹用于存放重命名后的文件
new_train_folder = "new_train"
new_train_labels_folder = "new_train_labels"

# 创建新文件夹
os.makedirs(new_train_folder, exist_ok=True)
os.makedirs(new_train_labels_folder, exist_ok=True)

# 遍历train文件夹中的图片
for i, filename in enumerate(sorted(os.listdir(train_folder))):

        new_filename = f"IMG_{i + 1}.jpg"
        old_filepath = os.path.join(train_folder, filename)
        new_filepath = os.path.join(new_train_folder, new_filename)

        # 重命名图片文件
        os.rename(old_filepath, new_filepath)

        print(f"将文件 {filename} 改名为 {new_filename}")

        # 构建对应的标注文件名
        old_label_filename = filename.split(".")[0] + ".xml"
        new_label_filename = f"GT_IMG_{i + 1}.xml"
        old_label_filepath = os.path.join(train_labels_folder, old_label_filename)
        new_label_filepath = os.path.join(new_train_labels_folder, new_label_filename)

        # 重命名标注文件
        os.rename(old_label_filepath, new_label_filepath)

        print(f"将文件 {old_label_filename} 改名为 {new_label_filename}")
