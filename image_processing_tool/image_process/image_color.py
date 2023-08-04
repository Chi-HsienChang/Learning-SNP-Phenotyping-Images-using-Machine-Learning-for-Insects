# import cv2
# import numpy as np

# image = cv2.imread('./output.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (7, 7), 0)
# edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
# binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)[1]
# cv2.imwrite('./output2.jpg', binary)


# import cv2

# # 讀取圖像
# img = cv2.imread('./RW-A-AR0302-03-14-19_M.jpg')

# # 將彩色圖像轉換為灰度圖像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 應用邊緣檢測，例如 Canny
# edges = cv2.Canny(gray, threshold1=30, threshold2=100)

# # 將結果保存到檔案
# cv2.imwrite('output.jpg', edges)

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Read the image
# image = cv2.imread('./RW-A-AR0302-03-14-19_M.jpg', cv2.IMREAD_GRAYSCALE)

# # Use Canny edge detection to find the contours of the image
# edges = cv2.Canny(image, threshold1=30, threshold2=100)
# cv2.imwrite('output.jpg', edges)

# from PIL import Image, ImageFilter
# import numpy as np

# # 讀取圖像並轉換為灰度
# img = Image.open('./RW-A-AR0302-03-14-19_M.jpg').convert('L')

# # 應用邊緣檢測
# edges = img.filter(ImageFilter.FIND_EDGES)

# # 轉換成 numpy 陣列
# edges_np = np.array(edges)

# # 進行二值化操作，閾值設定為 100
# edges_np = (edges_np > 100) * 255

# # 再轉換回 PIL 圖像
# edges = Image.fromarray(edges_np.astype(np.uint8))

# # 保存圖像
# edges.save('output.jpg')







#----------------------------
from PIL import Image

# 讀取圖像
img = Image.open('./output.jpg')

# 檢查圖像是否包含透明通道
if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):

    # 產生一個新的白色背景圖像
    background = Image.new('RGB', img.size, (255, 255, 255))

    # 合併背景和原始圖像
    background.paste(img, mask=img.split()[3])  # 3 是透明度通道
    img = background

# 儲存圖像
img.save('output2.jpg')

