#week 6
'''
在监管学习中，我们利用原始标签评价算法表现
在非监管学习中，我们没有这种原始标签来进行评价
我们用一个指标评价聚类算法，叫做剪影系数评分
它受当前数据点和聚类中其他数据点平均距离
和当前数据点和临近聚类中所有数据点平均距离影响
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import utilities

#载入数据
data = utilities.load_data('data_perf.txt')

scores = []
range_values = np.arange(2, 10)

for i in range_values:
#训练模型
	kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
	kmeans.fit(data)
	score = metrics.silhouette_score(data, kmeans.labels_, 
		metric='euclidean', sample_size=len(data))

	print ("\nNumber of clusters =", i)
	print ("Silhouette score =", score)

	scores.append(score)

#绘制分数
plt.figure()
plt.bar(range_values, scores, width=0.6, color='k', align='center')
plt.title('Silhouette score vs number of clusters')

#绘制数据
plt.figure()
plt.scatter(data[:,0], data[:,1], color='k', s=30, marker='o', facecolors='none')
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()