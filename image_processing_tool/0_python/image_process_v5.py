import yaml
from rembg import remove
import cv2
import numpy as np
import os

with open('user-defined.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)


Blur=7
L=30
H=100
angle=45
position=(0, 0)
maximum_diameter=5000

input_paths = ['./TEST1.jpg', './TEST2.jpg', './TEST3.jpg']  # Add the paths of the input images
output_folder = './output'  # Specify the output folder path

# Image matting
def M(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input_data = i.read()
            output_data = remove(input_data)
            o.write(output_data)

# Convert to black and white image
def B(input_path, output_path, Blur=7, L=30, H=100):
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (Blur, Blur), 0)
    edges = cv2.Canny(blurred, threshold1=L, threshold2=H)
    binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imwrite(output_path, binary)

# Rotate image
def R(input_path, output_path, angle=45, position=(0, 0)):
    img = cv2.imread(input_path)
    (height, width) = img.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    cv2.imwrite(output_path, rotated)

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
    # Generate output path based on the input image name
    file_name = os.path.basename(input_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_folder, file_name_without_extension)

    # Perform image processing steps
    if data['matting']['enable']:
        M(input_path, output_path + '_M.jpg')
        if data['black_and_white']['enable']:
            B(output_path + '_M.jpg', output_path + '_B.jpg', Blur, L, H)
            if data['rotation']['enable']:
                R(output_path + '_B.jpg', output_path + '_R.jpg', angle, position)
                if data['diameter']['enable']:
                    D(output_path + '_R.jpg', output_path + '_D.jpg', maximum_diameter) 
                    print(file_name, "processed (MBRD)") 
                else:
                    print(file_name, "processed (MBR)") 
            else:
                if data['diameter']['enable']:
                    D(output_path + '_B.jpg', output_path + '_D.jpg', maximum_diameter) 
                    print(file_name, "processed (MBD)") 
                else:
                    print(file_name, "processed (MB)") 
        else:
            if data['rotation']['enable']:
                R(output_path + '_M.jpg', output_path + '_R.jpg', angle, position)
                if data['diameter']['enable']:
                    D(output_path + '_R.jpg', output_path + '_D.jpg', maximum_diameter) 
                    print(file_name, "processed (MRD)") 
                else:
                    print(file_name, "processed (MR)") 
            else:
                if data['diameter']['enable']:
                    D(output_path + '_M.jpg', output_path + '_D.jpg', maximum_diameter) 
                    print(file_name, "processed (MD)") 
                else:
                    print(file_name, "processed (M)") 
    else:
        if data['black_and_white']['enable']:
            B(input_path, output_path + '_B.jpg', Blur, L, H)
            if data['rotation']['enable']:
                R(output_path + '_B.jpg', output_path + '_R.jpg', angle, position)
                if data['diameter']['enable']:
                    D(output_path + '_R.jpg', output_path + '_D.jpg', maximum_diameter)
                    print(file_name, "processed (BRD)") 
                else:
                    print(file_name, "processed (BR)") 
            else:
                if data['diameter']['enable']:
                    D(output_path + '_B.jpg', output_path + '_D.jpg', maximum_diameter) 
                    print(file_name, "processed (BD)") 
                else:
                    print(file_name, "processed (B)") 
        else:
            if data['rotation']['enable']:
                R(input_path, output_path + '_R.jpg', angle, position)
                if data['diameter']['enable']:
                    D(output_path + '_R.jpg', output_path + '_D.jpg', maximum_diameter)
                    print(file_name, "processed (RD)") 
                else:
                    print(file_name, "processed (R)") 
            else:
                if data['diameter']['enable']:
                    D(input_path, output_path + '_D.jpg', maximum_diameter) 
                    print(file_name, "processed (D)") 
                else:
                    print(file_name, "No image processing features are enabled.")
