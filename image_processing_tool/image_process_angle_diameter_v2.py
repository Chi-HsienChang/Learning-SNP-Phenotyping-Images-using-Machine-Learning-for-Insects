import yaml
from rembg import remove
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd

# load images
input_folder = './wing_image_M_B'
image_files = os.listdir(input_folder)  
input_paths = []
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)  
    if os.path.isfile(file_path): 
        input_paths.append(file_path)  
output_folder = './output' 



def D2(input_path, output_path, maximum_diameter):
    def find_max_diameter(img):
        _, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_diameter = 0
        max_i = max_j = 0
        for contour in contours:
            for i in range(len(contour)):
                for j in range(i+1, len(contour)):
                    dist = np.sqrt(np.sum((contour[i] - contour[j]) ** 2))
                    if dist > max_diameter:
                        max_diameter = dist
                        max_i = i
                        max_j = j

        return max_diameter, contour[max_i], contour[max_j]

    def rotate_to_horizontal(img, point1, point2):
        # Calculate the angle between the line and horizontal direction
        dx = point2[0][0] - point1[0][0]
        dy = point2[0][1] - point1[0][1]
        angle = np.arctan2(dy, dx)

        # Calculate the center of the image
        center = (img.shape[1] // 2, img.shape[0] // 2)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle * 180 / np.pi, 1)

        # Apply the rotation to the image
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return rotated_img

    def resize_image(img, scale_percent):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        new_dimensions = (width, height)
        resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)
        return resized_img

    # Read the image in grayscale mode
    img = cv2.imread(input_path, 0)
    
    # Find the max diameter and the two points forming it
    max_diameter, point1, point2 = find_max_diameter(img)
    
    # Rotate the image to make the line formed by the two points horizontal
    rotated_img = rotate_to_horizontal(img, point1, point2)
    
    # Calculate the scaling factor
    scale_percent = maximum_diameter / max_diameter * 100
    
    # Resize the image
    resized_img = resize_image(rotated_img, scale_percent)
    
    # Save the result
    cv2.imwrite(output_path, resized_img)

    img = Image.open(output_path)

    # 創建一個新的背景圖像，尺寸可以根據需要調整
    background = Image.new('RGB', (1000, 1000), (255, 255, 255))

    # 將原始圖像粘貼到背景上，您可以指定左上角的座標（在這裡是 (200, 200)）
    background.paste(img, (70, 130))

    # 儲存圖像
    background.save(output_path)


# Resize the image based on the user-defined maximum diameter of the object in the image
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the input
for input_path in input_paths:
    my_id = input_path.split('-')[2]
    data = pd.read_csv('RW_228_IDs.csv')
    row = data[data['ID'] == my_id]
    if len(row) > 0:
        lengthpixels = row['lengthpixels'].values[0]
        print(f"Lengthpixels: {lengthpixels}")
        D2(input_path, my_id + '.jpg', lengthpixels)
        print(file_name, "processed (D)") 
    else:
        print(f"No data found for ID {id}")




