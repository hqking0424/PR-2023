import os
import xml.etree.ElementTree as ET

# 指定train_xml文件夹的路径和train文件夹的路径
train_xml_folder = 'train_xml'
train_folder = 'train'

# 遍历train_xml文件夹中的XML文件
xml_files = os.listdir(train_xml_folder)
for xml_file in xml_files:
    xml_path = os.path.join(train_xml_folder, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_count = len(root.findall('object'))

    # 如果object元素个数小于60，则删除该XML文件和对应的图片
    if object_count < 60:
        image_name = os.path.splitext(xml_file)[0] + '.jpg'
        image_path = os.path.join(train_folder, image_name)

        # 删除XML文件
        os.remove(xml_path)
        print(f"Deleted {xml_file}")

        # 删除对应的图片文件
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted {image_name}")

print("Completed.")
