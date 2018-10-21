#week 6
'''
凝聚集群和分层聚类有关
分层聚类是借助树结构分割和组合数据
自底向上的分层聚类将每一个独立点看成一个聚类进行合并
这就叫做凝聚聚类
自上向下的分层聚类将数据看成一个大类进行分割
'''

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

#定义实现凝聚集群的功能函数
def perform_clustering(X, connectivity, title, num_clusters=3, linkage='ward'):
	plt.figure()
	model = AgglomerativeClustering(linkage=linkage, 
		connectivity=connectivity, n_clusters=num_clusters)
	model.fit(X)

	#提取标签
	labels = model.labels_

	#明确说明不同聚类的标记
	markers = '.vx'

	for i, marker in zip(range(num_clusters), markers):
	#迭代点用不同的标记绘制它们
		plt.scatter(X[labels==i, 0], X[labels==i, 1], s=50, 
			marker=marker, color='k', facecolors='none')

	plt.title(title)

'''
为了体现凝聚聚类的优势
我们需要在空间上有联系同时相聚位置很近的点上测试
让我们定义一个功能得到螺旋状的点集
'''
def get_spiral(t, noise_amplitude=0.5):
	r = t
	x = r * np.cos(t)
	y = r * np.sin(t)

	return add_noise(x, y, noise_amplitude)

'''
在上面的代码中，我们添加了一些噪点使它的不确定性增高
定义这个函数
'''
def add_noise(x, y, amplitude):
	X = np.concatenate((x, y))
	X += amplitude * np.random.randn(2, X.shape[1])
	return X.T

 #得到位与玫瑰曲线的点集
def get_rose(t, noise_amplitude=0.02):
	'''
	如果k是奇数，它将有k瓣
	否则它将有2k瓣
	'''
	k = 5
	r = np.cos(k*t) + 0.25 
	x = r * np.cos(t)
	y = r * np.sin(t)
	
	return add_noise(x, y, noise_amplitude)

#使它更多样化，加入内转迹线
def get_hypotrochoid(t, noise_amplitude=0):
	a, b, h = 10.0, 2.0, 4.0
	x = (a - b) * np.cos(t) + h * np.cos((a - b) / b * t) 
	y = (a - b) * np.sin(t) - h * np.sin((a - b) / b * t) 

	return add_noise(x, y, 0)

if __name__=='__main__':
	# 得到样本数据
	n_samples = 500
	np.random.seed(2)
	t = 2.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
	X = get_spiral(t)

	# 没有连接
	connectivity = None 
	perform_clustering(X, connectivity, 'No connectivity')

	# 创建k连接图表
	connectivity = kneighbors_graph(X, 10, include_self=False)
	perform_clustering(X, connectivity, 'K-Neighbors connectivity')

	plt.show()