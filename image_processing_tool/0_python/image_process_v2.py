import yaml
from rembg import remove
import cv2
import numpy as np
import os

with open('user-defined.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)



input_path = './TEST.jpg'
output_path = './TEST_MASK.jpg'
Blur=7
L=30
H=100
angle=45
position=(0, 0)
maximum_diameter=5000

# image matting 
def M(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

# Convert to black and white image
def B(input_path, output_path, Blur=7, L=30, H=100):
    image = cv2.imread('./TEST_MASK.JPG')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imwrite('./TEST_COLOR.JPG', binary)

# Rotate image
def R(input_path, output_path, angle=45, position=(0, 0)):
    img = cv2.imread('./TEST_COLOR.JPG')
    (height, width) = img.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    cv2.imwrite('./TEST_ROTATE.JPG', rotated)

# Resize the image based on the user-defined maximum diameter of the object in the image
def D(input_path, output_path, maximum_diameter=5000):
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

    img = cv2.imread('./TEST_ROTATE.JPG', cv2.IMREAD_GRAYSCALE)
    max_diameter = find_max_diameter(img)
    target_diameter = 6000 
    scale_percent = (target_diameter / max_diameter) * 100
    resized_img = resize_image(img, scale_percent)
    cv2.imwrite('./TEST_DIAMETER_6000.JPG', resized_img)


if data['matting']['enable']:
    M(input_path, output_path)
    if data['black_and_white']['enable']:
        B(input_path, output_path, Blur, L, H)
        if data['rotation']['enable']:
            R(input_path, output_path, angle, position)
            if data['diameter']['enable']:
                D(input_path, output_path, maximum_diameter) 
                print("MBRD") # MBRD
            else:
                print("MBR") # MBR
        else:
            if data['diameter']['enable']:
                D(input_path, output_path, maximum_diameter) 
                print("MBD") # MBD
            else:
                print("MB") # MB
    else:
        if data['rotation']['enable']:
            R(input_path, output_path, angle, position)
            if data['diameter']['enable']:
                D(input_path, output_path, maximum_diameter) 
                print("MRD") # MRD
            else:
                print("MR") # MR
        else:
            if data['diameter']['enable']:
                D(input_path, output_path, maximum_diameter) 
                print("MD") # MD
            else:
                print("M") # M
else:
    if data['black_and_white']['enable']:
        B(input_path, output_path, Blur, L, H)
        if data['rotation']['enable']:
            R(input_path, output_path, angle, position)
            if data['diameter']['enable']:
                D(input_path, output_path, maximum_diameter)
                print("BRD") # BRD
            else:
                print("BR") # BR
        else:
            if data['diameter']['enable']:
                D(input_path, output_path, maximum_diameter) 
                print("BD") # BD
            else:
                print("B") # B        
    else:
        if data['rotation']['enable']:
            R(input_path, output_path, angle, position)
            if data['diameter']['enable']:
                D(input_path, output_path, maximum_diameter)
                print("RD") # RD
            else:
                print("R") # R
        else:
            if data['diameter']['enable']:
                D(input_path, output_path, maximum_diameter) 
                print("D") # D
            else:
                print("No image processing features are enabled.") # None





