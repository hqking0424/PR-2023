import os
import csv
import xml.etree.ElementTree as ET
import cv2

# 指定train文件夹的路径和train.csv文件的路径
train_folder = 'train'
train_csv = 'train.csv'

# 创建train_xml文件夹用于存放bbox标注信息的XML文件
if not os.path.exists('train_xml'):
    os.makedirs('train_xml')

# 读取train.csv中的编号和bbox标注信息
with open(train_csv, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # 跳过标题行
    bbox_dict = {}  # 存储每个编号对应的bbox标注信息
    for row in csv_reader:
        image_id = row[0]
        bbox = row[3]

        # 如果编号不存在于字典中，则创建新的键值对
        if image_id not in bbox_dict:
            bbox_dict[image_id] = []

        # 将bbox标注信息添加到对应编号的列表中
        bbox_dict[image_id].append(bbox)

# 遍历train文件夹中的图片，并将bbox标注信息写入XML文件
for image_name in os.listdir(train_folder):
    image_id = os.path.splitext(image_name)[0]
    image_path = os.path.join(train_folder, image_name)

    # 如果编号存在于字典中，则将bbox标注信息写入XML文件
    if image_id in bbox_dict:
        xml_path = os.path.join('train_xml', f"{image_id}.xml")

        # 创建根元素 <annotation>
        root = ET.Element("annotation")
        root.set("verified", "no")

        # 创建子元素 <folder>、<filename>、<path>
        folder = ET.SubElement(root, "folder")
        folder.text = "date2"

        filename = ET.SubElement(root, "filename")
        filename.text = image_id

        path = ET.SubElement(root, "path")
        path.text = image_path

        # 读取图像尺寸
        img = cv2.imread(image_path)
        height, width, channels = img.shape

        # 创建子元素 <size>
        size = ET.SubElement(root, "size")
        width_elem = ET.SubElement(size, "width")
        height_elem = ET.SubElement(size, "height")
        depth_elem = ET.SubElement(size, "depth")
        width_elem.text = str(width)
        height_elem.text = str(height)
        depth_elem.text = str(channels)

        # 创建子元素 <segmented>
        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"

        # 遍历bbox标注信息，为每个bbox创建子元素 <object>
        for bbox in bbox_dict[image_id]:
            bbox = bbox.strip('[]').split(',')
            bbox = [float(coord) for coord in bbox]

            obj = ET.SubElement(root, "object")

            name = ET.SubElement(obj, "name")
            name.text = "ear"  # 根据实际对象名称填写

            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"

            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            ymin = ET.SubElement(bndbox, "ymin")
            xmax = ET.SubElement(bndbox, "xmax")
            ymax = ET.SubElement(bndbox, "ymax")
            xmin.text = str(int(bbox[0]))
            ymin.text = str(int(bbox[1]))
            xmax.text = str(int(bbox[0] + bbox[2]))
            ymax.text = str(int(bbox[1] + bbox[3]))

        # 创建XML树并写入文件
        tree = ET.ElementTree(root)
        tree.write(xml_path)

        print(f"Processed image {image_name}")

print("Completed.")
