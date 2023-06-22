import os
import xml.etree.ElementTree as ET

# 定义文件夹路径
train_labels_folder = "train_labels"
train_txt_folder = "train_txt"

# 创建新的文件夹用于存放转换后的文件
os.makedirs(train_txt_folder, exist_ok=True)

# 遍历train_labels文件夹中的XML文件
for filename in sorted(os.listdir(train_labels_folder)):
    if filename.endswith(".xml"):
        xml_filepath = os.path.join(train_labels_folder, filename)
        txt_filepath = os.path.join(train_txt_folder, os.path.splitext(filename)[0] + ".txt")

        # 解析XML文件
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        with open(txt_filepath, "w") as txt_file:
            # 遍历每个<object>标签
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")

                # 获取边界框坐标
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # 计算中心点坐标
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2

                # 将结果写入TXT文件
                txt_file.write(f"{x_center} {y_center}\n")

        print(f"将文件 {filename} 转换为 {os.path.splitext(filename)[0]}.txt")
