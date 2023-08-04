import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 0.5  # 平均值
sigma = 0.1  # 標準差

# 定義三個範圍以及對應的顏色和標籤
ranges = [(0, 2, 'black', 't=0'), (1, 3, 'red', 't=1'), (2, 4, 'blue', 't=2')]

plt.figure(figsize=(10,6))

for range_ in ranges:
    x = np.linspace(range_[0], range_[1], 100)
    y = stats.norm.pdf(x, np.mean(range_[:2]), sigma)
    
    # Normalizing y to have max value of 1
    y = y / np.max(y)

    plt.plot(x, y, color=range_[2])
    max_y_index = np.argmax(y)
    plt.text(x[max_y_index] - 0.12, y[max_y_index] + 0.1, range_[3], fontsize=22, color=range_[2])  # 對齊左邊

plt.xlabel('Reward', fontsize=22)  # 改變 x 軸標籤的字體大小
plt.ylabel('Probability density', fontsize=22)  # 改變 y 軸標籤的字體大小
plt.grid(True)
plt.ylim(0, 1.2)  # 設定 y 軸的範圍

# 將圖表保存為圖片
plt.savefig('single_figure_normal_distributions.png', dpi=300)

plt.show()
