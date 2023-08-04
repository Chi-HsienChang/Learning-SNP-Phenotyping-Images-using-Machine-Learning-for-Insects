import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 0.5  # 平均值
sigma = 0.1  # 標準差

# 生成 x 的值
x = np.linspace(0, 1, 100)

# 計算對應的 y 的值，即概率密度
y = stats.norm.pdf(x, mu, sigma)

# 使 y 軸的最大值為 1
y = y / np.max(y)

plt.figure(figsize=(10,6))
plt.axis(False)
plt.plot(x, y, color='red')  # 指定線條顏色為紅色

plt.ylim(0, 1.2)

# 將圖表保存為圖片
plt.savefig('normal_distribution.png', dpi=300)

plt.show()
