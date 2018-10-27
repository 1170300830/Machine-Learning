#week 7
'''
k-nearest neighbors 是一种用k-最近领域训练数据和发现一个未知个体类的算法
当我们想要知道一个未知点属于什么类时
我们寻找k-最近领域并分析这个点属于各个领域的可能性
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors, datasets

from utilities import load_data

#载入输入数据
input_file = 'data_nn_classifier.txt'
data = load_data(input_file)
X, y = data[:,:-1], data[:,-1].astype(np.int)
'''
前两列包含载入数据
后一列包含标签
'''

#绘制输入数据
#遍历数据集用合适的符号区别数据类
plt.figure()
plt.title('Input datapoints')
markers = '^sov<>hp'
mapper = np.array([markers[i] for i in y])
for i in range(X.shape[0]):
	plt.scatter(X[i, 0], X[i, 1], marker=mapper[i],
		s=50, edgecolors='black', facecolors='none')

#设置领域的数目以创建分类
num_neighbors = 10

#定义网格并计算在网格中的分类
h = 0.01

#创建k-最近领域分类器模型并训练它
classifier = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
classifier.fit(X, y)

#创建网格，绘制边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#估算所有点的分类输出
predicted_values = classifier.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

#绘制分类器的结果
predicted_values = predicted_values.reshape(x_grid.shape)
plt.figure()
plt.pcolormesh(x_grid, y_grid, predicted_values, cmap=cm.Pastel1)

#将点集覆盖在分类边界上观察分类效果
for i in range(X.shape[0]):
	plt.scatter(X[i, 0], X[i, 1], marker=mapper[i],
		s=50, edgecolors='black', facecolors='none')

plt.xlim(x_grid.min(), x_grid.max())
plt.ylim(y_grid.min(), y_grid.max())
plt.title('k nearest neighbors classifier boundaries')

#用测试点集测试分类器效果
test_datapoint = [4.5, 3.6]
plt.figure()
plt.title('Test datapoint')
for i in range(X.shape[0]):
	plt.scatter(X[i, 0], X[i, 1], marker=mapper[i],s=50,
		edgecolors='black', facecolors='none')

plt.scatter(test_datapoint[0], test_datapoint[1], marker='x',
	linewidth=3, s=200, facecolors='black')

#提取k最近领域
dist, indices = classifier.kneighbors([test_datapoint])

#绘制k最近领域
plt.figure()
plt.title('k nearest neighbors')

for i in indices:
	plt.scatter(X[i, 0], X[i, 1], marker='o',
		linewidth=3, s=100, facecolors='black')

plt.scatter(test_datapoint[0], test_datapoint[1], 
	marker='x',linewidth=3, s=200, facecolors='black')

for i in range(X.shape[0]):
	plt.scatter(X[i, 0], X[i, 1], marker=mapper[i],
		s=50, edgecolors='black', facecolors='none')

print ("Predicted output:", classifier.predict([test_datapoint])[0])

plt.show()