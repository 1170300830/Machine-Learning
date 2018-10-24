#week 7
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

#构造一些简单的二维数据
X = np.array([[1, 1], [1, 3], [2, 2], [2.5, 5], [3, 1],
	[4, 2], [2, 3.5], [3, 3], [3.5, 4]])

#发现给出点的最近的三个相邻点
num_neighbors = 3

#定义一个没有出现在输入点中的随机点
input_point = [2.6, 1.7]

#绘制数据点
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', s=25, color='k')

#构建最近邻域模型
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
distances, indices = knn.kneighbors([input_point])

#输出最近的领域
print ("\nk nearest neighbors")
for rank, index in enumerate(indices[0][:num_neighbors]):
	print (str(rank+1) + " -->", X[index])

#绘制最近的领域
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', s=25, color='k')
plt.scatter(X[indices][0][:][:,0], X[indices][0][:][:,1],
	marker='o', s=150, color='k', facecolors='none')
plt.scatter(input_point[0], input_point[1],
	marker='o', s=150, color='red', facecolors='none')

plt.show()