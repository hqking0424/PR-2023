import os

# 指定train文件夹的路径和train_xml文件夹的路径
train_folder = 'train'
train_xml_folder = 'train_xml'

# 遍历train文件夹中的图片
image_files = os.listdir(train_folder)
for image_file in image_files:
    image_name = os.path.splitext(image_file)[0]
    xml_file = image_name + '.xml'
    xml_path = os.path.join(train_xml_folder, xml_file)

    # 如果对应的XML文件不存在，则删除该图片文件
    if not os.path.exists(xml_path):
        image_path = os.path.join(train_folder, image_file)
        os.remove(image_path)
        print(f"Deleted {image_file}")

print("Completed.")
