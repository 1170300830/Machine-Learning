#week 6
'''
平均移位是一种无监管学习下聚集数据点的有力算法
它从概率密度的角度看待数据点
平均移位算法的主要优势是我们在处理前无需知道群组的数量
'''

import numpy as np 
from sklearn.cluster import MeanShift,estimate_bandwidth
import utilities

#加载输入数据
X = utilities.load_data('data_multivar.txt')

#估算带宽
bandwidth = estimate_bandwidth(X,quantile = 0.1,n_samples =len(X))

#使用平均位移计算聚类
meanshift_estimator = MeanShift(bandwidth = bandwidth,bin_seeding = True)

#训练模型
meanshift_estimator.fit(X)

#提取标签
labels = meanshift_estimator.labels_

#分析聚类的质心，打印聚类的数量
centroids = meanshift_estimator.cluster_centers_
num_clusters = len(np.unique(labels))

print("Number of clusters in input data =",num_clusters)

#绘制点和质心
import matplotlib.pyplot as plt 
from itertools import cycle

plt.figure()

#明确不同聚类的标记
markers = '.*xv'

#遍历点并绘制
for i, marker in zip(range(num_clusters),markers):
	#绘制属于当前集群的点
	plt.scatter(X[labels==i,0],X[labels==i,1],marker = marker,color ='k')

	#绘制当前集群点的质心
	centroid = centroids[i]
	plt.plot(centroid[0],centroid[1],marker = 'o',
		markerfacecolor = 'k',markeredgecolor = 'k',
		markersize = 15)
plt.title('Clusters and their centroids')
plt.show()