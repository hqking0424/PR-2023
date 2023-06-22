import os

train_folder = 'train'
train_txt_folder = 'train_txt'

# 获取train文件夹中的图片文件列表
image_files = os.listdir(train_folder)

# 获取train_txt文件夹中的标注文件列表
txt_files = os.listdir(train_txt_folder)

# 图片文件编号计数器
image_count = 166

# 遍历图片文件并重新编号
print("改名记录:")
for image_file in image_files:
    image_name, ext = os.path.splitext(image_file)
    image_new_name = f'IMG_{image_count}{ext}'
    image_count += 1
    image_path = os.path.join(train_folder, image_file)
    new_image_path = os.path.join(train_folder, image_new_name)
    os.rename(image_path, new_image_path)
    print(f'{image_file} -> {image_new_name}')

    # 找到对应的标注文件并改名
    txt_file = f'GT_{image_name}.txt'
    if txt_file in txt_files:
        txt_path = os.path.join(train_txt_folder, txt_file)
        new_txt_file = f'GT_IMG_{image_count - 1}.txt'
        new_txt_path = os.path.join(train_txt_folder, new_txt_file)
        os.rename(txt_path, new_txt_path)
        print(f'{txt_file} -> {new_txt_file}')

print("改名完成。")
