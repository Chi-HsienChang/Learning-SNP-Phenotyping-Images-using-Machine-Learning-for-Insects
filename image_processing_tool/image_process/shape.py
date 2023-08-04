import cv2

# Load the image
img = cv2.imread('./TEST_DIAMETER_5000.JPG')

# Check if image is loaded successfully
if img is not None:
    # Print the shape of the image
    print('Image Shape:', img.shape)
else:
    print("Image not loaded. Please check the path and try again.")


img = cv2.imread('./TEST_DIAMETER_8000.JPG')

# Check if image is loaded successfully
if img is not None:
    # Print the shape of the image
    print('Image Shape:', img.shape)
else:
    print("Image not loaded. Please check the path and try again.")