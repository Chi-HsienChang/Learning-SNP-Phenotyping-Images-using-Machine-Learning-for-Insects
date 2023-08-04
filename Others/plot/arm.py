import numpy as np
import matplotlib.pyplot as plt

# 生成一個網格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# 定義兩個不同的高斯分佈
Z1 = np.exp(-((X-1)**2 + (Y-1)**2))
Z2 = np.exp(-((X+1)**2 + (Y+1)**2))

# 將兩個高斯分佈相加以形成兩個山峰
Z = Z1 + Z2

# 畫出等高線圖，這裡我們將所有等高線的顏色設置為紅色
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, 10, colors='black')
plt.clabel(contour, inline=True, fontsize=8)

# 在等高線圖內繪製五個不同顏色且較大的點
colors = ['red', 'blue', 'green', 'yellow', 'purple']
points = [(-1.5, -0.7), (0.8, 1), (-1, -0.8), (1, 0), (2, 2)]

for color, point in zip(colors, points):
    plt.scatter(*point, color=color, s=100, edgecolors='black', zorder=10)

# 繪製箭頭
plt.arrow(-2, -0.5, points[0][0]+2, points[0][1]+0.5, color='black', width=0.04, length_includes_head=True)

# 繪製紅色虛線圓形
circle = plt.Circle((-2, -0.5), 0.08, edgecolor='red', facecolor='None', linestyle='dotted')
plt.gca().add_patch(circle)

# plt.title('Contour Plot')
plt.xlabel('x')
plt.ylabel('y')

# 將圖表保存為圖片
plt.savefig('contour_plot.png', dpi=300)

plt.show()
