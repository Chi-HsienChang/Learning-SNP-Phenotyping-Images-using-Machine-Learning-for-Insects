import yaml
from rembg import remove
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd

# load images
input_folder = './test'
image_files = os.listdir(input_folder)  
input_paths = []
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)  
    if os.path.isfile(file_path): 
        input_paths.append(file_path)  
output_folder = './output' 


# Rotate image
def R(input_path, output_path, angle):
    img = cv2.imread(input_path)
    (height, width) = img.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    cv2.imwrite(output_path, rotated)

# Resize the image based on the user-defined maximum diameter of the object in the image
def D(input_path, output_path, maximum_diameter):
    def find_max_diameter(img):
        _, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_diameter = 0
        for contour in contours:
            for i in range(len(contour)):
                for j in range(i+1, len(contour)):
                    dist = np.sqrt(np.sum((contour[i] - contour[j]) ** 2))
                    max_diameter = max(max_diameter, dist)
        return max_diameter

    def resize_image(img, scale_percent):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        new_dimensions = (width, height)
        resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)
        return resized_img

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    max_diameter = find_max_diameter(img)
    target_diameter = 6000
    scale_percent = (target_diameter / max_diameter) * 100
    resized_img = resize_image(img, scale_percent)
    cv2.imwrite(output_path, resized_img)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the input
for input_path in input_paths:
    img = Image.open(input_path)

    # 創建一個新的背景圖像，尺寸可以根據需要調整
    background = Image.new('RGB', (5000, 5000), (255, 255, 255))

    # 將原始圖像粘貼到背景上，您可以指定左上角的座標（在這裡是 (200, 200)）
    background.paste(img, (200, 200))

    # 儲存圖像
    background.save(input_path)

    my_id = input_path.split('-')[2]
    data = pd.read_csv('RW_228_IDs.csv')
    row = data[data['ID'] == my_id]
    if len(row) > 0:
        lengthpixels = row['lengthpixels'].values[0]
        print(f"Lengthpixels: {lengthpixels}")
        D(input_path, my_id + '.jpg', lengthpixels)
        print(file_name, "processed (D)") 
    else:
        print(f"No data found for ID {id}")


    # file_name_without_extension = os.path.splitext(file_name)[0]
    # output_path = os.path.join(output_folder, file_name_without_extension)

    # R(input_path, output_path + '_R.{}'.format(data['output_format']), data['rotation']['angle'])
    # D(output_path + '_R.{}'.format(data['output_format']), output_path + '_D.{}'.format(data['output_format']), data['diameter']['maximum_diameter'])
    # print(file_name, "processed (RD)") 



# import pandas as pd

# # 從檔案名稱中提取 ID
# filename = "RW-A-AR0302-03-14-19_M_B.jpg"
# id = filename.split('-')[2]

# # 從 CSV 檔案中讀取資料
# data = pd.read_csv('data.csv')

# # 搜尋相同 ID 的資料
# row = data[data['ID'] == id]

# # 檢查是否找到相同 ID 的資料
# if len(row) > 0:
#     # 獲取 Angle 和 Lengthpixels
#     angle = row['Angle'].values[0]
#     lengthpixels = row['lengthpixels'].values[0]
#     print(f"Angle: {angle}, Lengthpixels: {lengthpixels}")
# else:
#     print(f"No data found for ID {id}")

