#week 6
'''
在k-means算法中，我们必须给出聚类数量
在现实世界中，我们很多时候没有这个信息
我们可以用剪影系数评分得到聚类数量
但是这是一个代价很高的处理方案
DBSCAN 是一个代价较小得到聚类数量的方案
我们控制点与其他点的最大距离
这个方法的一大优势是可以处理异常
假如一些点出现在低密度区域，DBSCAN可以识别并强制它们聚类
'''

from itertools import cycle

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

from utilities import load_data

#加载数据
input_file = 'data_perf.txt'
X = load_data(input_file)

#发现最佳 epsilon
eps_grid = np.linspace(0.3, 1.2, num=10)
silhouette_scores = []
eps_best = eps_grid[0]
silhouette_score_max = -1
model_best = None
labels_best = None
for eps in eps_grid:
	#训练DBSCAN模型
	model = DBSCAN(eps=eps, min_samples=5).fit(X)

	#提取标签
	labels = model.labels_

	#提取最佳参数
	silhouette_score = round(metrics.silhouette_score(X, labels), 4)
	silhouette_scores.append(silhouette_score)

	print ("Epsilon:", eps, " --> silhouette score:", silhouette_score)

	if silhouette_score > silhouette_score_max:
		silhouette_score_max = silhouette_score
		eps_best = eps
		model_best = model
		labels_best = labels

#绘制剪影系数分数和epsilon对比图
plt.figure()
plt.bar(eps_grid, silhouette_scores, width=0.05, color='k', align='center')
plt.title('Silhouette score vs epsilon')

#最佳参数
print ("\nBest epsilon =", eps_best)

#关联模型和标签
model = model_best 
labels = labels_best

#识别未分配的点
offset = 0
if -1 in labels:
	offset = 1

#提取聚类的数字
num_clusters = len(set(labels)) - offset 
print ("\nEstimated number of clusters =", num_clusters)

#训练训练模型的核心样本
mask_core = np.zeros(labels.shape, dtype=np.bool)
mask_core[model.core_sample_indices_] = True

#绘制输出聚类
plt.figure()
labels_uniq = set(labels)
markers = cycle('vo^s<>')
for cur_label, marker in zip(labels_uniq, markers):
#用黑点表示未分配数据
	if cur_label == -1:
		marker = '.'

	#为当前点设置标记
	cur_mask = (labels == cur_label)

	cur_data = X[cur_mask & mask_core]
	plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker,
		edgecolors='black', s=96, facecolors='none')

	cur_data = X[cur_mask & ~mask_core]
	plt.scatter(cur_data[:, 0], cur_data[:, 1], marker=marker,
		edgecolors='black', s=32)

plt.title('Data separated into clusters')
plt.show()