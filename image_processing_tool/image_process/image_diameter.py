import cv2
import numpy as np

def find_max_diameter(img):
    # Threshold the image to binary (you may need to adjust the threshold value)
    _, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the maximum diameter among the contours
    max_diameter = 0
    for contour in contours:
        for i in range(len(contour)):
            for j in range(i+1, len(contour)):
                # Calculate distance between two points
                dist = np.sqrt(np.sum((contour[i] - contour[j]) ** 2))
                max_diameter = max(max_diameter, dist)

    return max_diameter

def resize_image(img, scale_percent):
    # Compute new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)

    # Resize the image
    resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)

    return resized_img

# Load the image
img = cv2.imread('./TEST_ROTATE.JPG', cv2.IMREAD_GRAYSCALE)

# Find the maximum diameter of the colored pixels in the image
max_diameter = find_max_diameter(img)

# Specify a target maximum diameter
target_diameter = 8000  # adjust this value according to your requirement

# Calculate the scale percent required to resize the image
scale_percent = (target_diameter / max_diameter) * 100

# Resize the image to the target maximum diameter
resized_img = resize_image(img, scale_percent)

# Write it back to disk
cv2.imwrite('./TEST_DIAMETER_8000.JPG', resized_img)




# ./TEST_ROTATE.JPG
# ./TEST_DIAMETER.JPG