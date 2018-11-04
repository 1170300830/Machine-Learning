#week 8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

#获得样本数据
amplitude = 10
num_points = 100
X = amplitude * np.random.rand(num_points, 1) - 0.5 * amplitude

#计算目标及添加噪点
y = np.sinc(X).ravel() 
y += 0.2 * (0.5 - np.random.rand(y.size))

#绘制输入图像
plt.figure()
plt.scatter(X, y, s=40, c='k', facecolors='none')
plt.title('Input data')

#创造一个十倍输入密度的一维网格
x_values = np.linspace(-0.5*amplitude, 0.5*amplitude, 10*num_points)[:, np.newaxis]
'''
使用更密集的网格是为了用所有点来评估回归器
观察它近似函数的效果
'''

#规定最近领域的数量
n_neighbors = 8

#定义和训练回归器
knn_regressor = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_values = knn_regressor.fit(X, y).predict(x_values)

#重叠输入数据和回归结果
plt.figure()
plt.scatter(X, y, s=40, c='k', facecolors='none', label='input data')
plt.plot(x_values, y_values, c='k', linestyle='--', label='predicted values')
plt.xlim(X.min() - 1, X.max() + 1)
plt.ylim(y.min() - 0.2, y.max() + 0.2)
plt.axis('tight')
plt.legend()
plt.title('K Nearest Neighbors Regressor')

plt.show()

'''
回归器的目的是预测连续的输出值
我们没有一个固定的输出值的类别数
在这个例子中
我们用sinc函数展示k-nearest-neighbor算法
sinc(x) = sin(x)/x
'''