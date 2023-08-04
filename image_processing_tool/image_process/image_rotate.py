import cv2
import numpy as np
img = cv2.imread('./TEST_COLOR.JPG')
(height, width) = img.shape[:2]
center = (width / 2, height / 2)
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
cv2.imwrite('./TEST_ROTATE.JPG', rotated)