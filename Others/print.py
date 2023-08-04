from PIL import Image

# 開啟圖片
img = Image.open('./wing_image_for_ML_B_and_W/AR0101.jpg')

# 獲取並印出圖片大小
width, height = img.size
print(f'圖片寬度: {width}px, 圖片高度: {height}px')
