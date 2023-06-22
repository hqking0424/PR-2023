import os
import shutil

# 指定train文件夹的路径和train_xml文件夹的路径
train_folder = 'train'
train_xml_folder = 'train_xml'

# 创建train_renamed文件夹用于存放重新编号后的图片文件
if not os.path.exists('train_renamed'):
    os.makedirs('train_renamed')
# 创建train_xml_renamed文件夹用于存放重新编号后的xml文件
if not os.path.exists('train_xml_renamed'):
    os.makedirs('train_xml_renamed')


# 遍历train文件夹中的图片
image_files = os.listdir(train_folder)
for i, image_file in enumerate(image_files):
    old_image_path = os.path.join(train_folder, image_file)
    new_image_name = str(i + 1) + '.jpg'
    new_image_path = os.path.join('train_renamed', new_image_name)

    # 将图片重新命名并复制到train_renamed文件夹中
    shutil.copyfile(old_image_path, new_image_path)

    # 查找对应的XML标注文件
    old_xml_name = os.path.splitext(image_file)[0] + '.xml'
    old_xml_path = os.path.join(train_xml_folder, old_xml_name)
    new_xml_name = str(i + 1) + '.xml'
    new_xml_path = os.path.join('train_xml_renamed', new_xml_name)

    # 修改XML标注文件的名字并复制到train_xml_renamed文件夹中
    shutil.copyfile(old_xml_path, new_xml_path)

    print(f"Renamed {image_file} to {new_image_name}")
    print(f"Renamed {old_xml_name} to {new_xml_name}")

print("Completed.")
