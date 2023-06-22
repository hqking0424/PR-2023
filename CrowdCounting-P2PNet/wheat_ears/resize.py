import os
import cv2

# 图片文件夹路径
image_folder = 'test'
# 标注文件夹路径
label_folder = 'test_txt'

# 遍历图片文件夹
for i in range(1, 72):
    image_filename = f'IMG_{i}.jpg'
    label_filename = f'GT_IMG_{i}.txt'

    image_path = os.path.join(image_folder, image_filename)
    label_path = os.path.join(label_folder, label_filename)

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        continue

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # 根据图片尺寸进行调整
    if width == 6000 and height == 4000:
        new_width, new_height = 1280, 720
        x_scale = new_width / width
        y_scale = new_height / height
    elif width == 4000 and height == 6000:
        new_width, new_height = 720, 1280
        x_scale = new_width / width
        y_scale = new_height / height
    else:
        continue

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path, resized_image)

    # 处理对应的标注文件
    with open(label_path, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            coords = line.strip().split()
            x = float(coords[0]) * x_scale
            y = float(coords[1]) * y_scale
            f.write(f'{x} {y}\n')
        f.truncate()

    print(f"将{image_filename}从{width}*{height}改为{new_width}*{new_height}，"
          f"将{label_filename}按照x轴{x_scale}比例、y轴{y_scale}比例进行缩放")
