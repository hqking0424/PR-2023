import os

# 文件夹路径
image_folder = "./test"
txt_folder = "./test_txt"

# 获取文件夹中的文件列表
image_files = sorted(os.listdir(image_folder))
txt_files = sorted(os.listdir(txt_folder))

# 检查文件数量是否匹配
if len(image_files) != len(txt_files):
    print("文件数量不匹配")
    exit(1)

# 创建 train.txt 文件并写入路径
with open("new_wheat_test.list", "w") as f:
    for i in range(len(image_files)):
        image_path = os.path.join(image_folder, image_files[i])
        txt_path = os.path.join(txt_folder, txt_files[i])
        f.write(f"{image_path} {txt_path}\n")
