import os

train_folder = 'test'
train_txt_folder = 'test_txt'

# 遍历train_txt文件夹下的每个txt文件
for txt_file in os.listdir(train_txt_folder):
    if txt_file.startswith('GT_IMG_') and txt_file.endswith('.txt'):
        txt_path = os.path.join(train_txt_folder, txt_file)
        image_number = int(txt_file.split('_')[2].split('.')[0])

        # 检查txt文件的行数
        with open(txt_path, 'r') as file:
            line_count = sum(1 for _ in file)

        # 如果行数小于20，则删除txt文件和对应的图片
        if line_count < 20:
            os.remove(txt_path)
            image_file = f'IMG_{image_number}.jpg'
            image_path = os.path.join(train_folder, image_file)
            os.remove(image_path)
            print(f'Deleted {txt_file} and corresponding image {image_file}')
