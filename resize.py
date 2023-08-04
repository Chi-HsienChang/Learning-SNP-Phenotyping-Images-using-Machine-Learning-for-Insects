import os
from PIL import Image

# 定義要調整的大小
target_size = (200, 200)

# 輸入和輸出資料夾路徑
input_folder = './wing_image_for_ML_gray'
output_folder = './wing_image_resized'

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 取得輸入資料夾下所有圖片的檔案路徑
image_files = os.listdir(input_folder)

for img_file in image_files:
    # 讀取圖片
    img_path = os.path.join(input_folder, img_file)
    image = Image.open(img_path)

    # 調整大小並儲存
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    output_path = os.path.join(output_folder, img_file)
    resized_image.save(output_path)

print("圖片調整大小並儲存完成！")
