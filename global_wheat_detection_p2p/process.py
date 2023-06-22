SSimport os
import csv

# 指定train文件夹的路径和train.csv文件的路径
train_folder = 'train'
train_csv = 'train.csv'

# 创建train_txt文件夹用于存放bbox标注信息的文本文件
if not os.path.exists('train_txt'):
    os.makedirs('train_txt')

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

# 遍历train文件夹中的图片，并将bbox标注信息写入txt文件
for image_name in os.listdir(train_folder):
    image_id = os.path.splitext(image_name)[0]

    # 如果编号存在于字典中，则将bbox标注信息写入txt文件
    if image_id in bbox_dict:
        txt_path = os.path.join('train_txt', f"{image_id}.txt")
        with open(txt_path, 'w') as txt_file:
            for bbox in bbox_dict[image_id]:
                bbox = bbox.strip('[]').split(',')
                bbox = [float(coord) for coord in bbox]
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                txt_file.write(f"{int(x_center)} {int(y_center)}\n")

        print(f"Processed image {image_name}")

print("Completed.")
