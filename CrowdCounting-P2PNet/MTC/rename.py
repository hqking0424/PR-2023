import os
import shutil

# 定义源文件夹和目标文件夹路径
image_folder = 'test/'
label_folder = 'test_txt/'
output_folder = 'output/'

# 创建目标文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历源文件夹中的图片文件
for idx, image_file in enumerate(sorted(os.listdir(image_folder))):
    # 构建新的图片文件名和标注文件名
    new_image_name = f'IMG_{idx + 1}.jpg'
    new_label_name = f'GT_IMG_{idx + 1}.txt'
    label_file = image_file.replace('.jpg', '.txt')
    label_name = f'GT_{label_file}'



    # 构建源文件和目标文件的完整路径
    src_image_path = os.path.join(image_folder, image_file)
    dest_image_path = os.path.join(output_folder, new_image_name)
    src_label_path = os.path.join(label_folder, label_name)
    dest_label_path = os.path.join(output_folder, new_label_name)

    # 将图片文件重命名并复制到目标文件夹中
    shutil.copyfile(src_image_path, dest_image_path)

    # 将标注文件重命名并复制到目标文件夹中
    shutil.copyfile(src_label_path, dest_label_path)

    # 打印输出每次重命名的信息
    print(f'Renamed {image_file} to {new_image_name}')
    print(f'Renamed {src_label_path} to {new_label_name}')
