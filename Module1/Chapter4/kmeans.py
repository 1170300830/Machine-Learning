#week 5
'''
无监督学习是一种不标记数据机器学习范式
这些算法根据相似度量将数据分成小团体
无监督学习一个最常用的方法是聚类
聚类常常依据相似特征例如欧几里得距离
无监督学习应用广泛，例如数据挖掘，医学成像等
'''

'''
k-means 算法是最著名的聚类算法之一
这个算法通过数据的各种属性将数据分成k个小团体
'''

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.cluster import KMeans 
import utilities

#加载数据
data = utilities.load_data('data_multivar.txt')
num_clusters = 4

plt.figure()
plt.scatter(data[:,0],data[:,1],marker = 'o',facecolors = 'none',edgecolors = 'k',s = 30)
x_min,x_max = min(data[:,0])-1,max(data[:,0])+1
y_min,y_max = min(data[:,1])-1,max(data[:,1])+1

plt.title('Input data')
plt.xlim(x_min,x_max)
plt.ylim(y_min,x_max)
plt.xticks(())
plt.yticks(())

#初始化和训练模型
kmeans = KMeans(init = 'k-means++',n_clusters = num_clusters,n_init = 10)
kmeans.fit(data)

#可视化边界
#设置网格大小
step_size = 0.01

#绘制边界
x_min,x_max = min(data[:,0])-1,max(data[:,0])+1
y_min,y_max = min(data[:,1])-1,max(data[:,1])+1
x_values,y_values = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))

#预测所有在网格内点的标签
predicted_labels = kmeans.predict(np.c_[x_values.ravel(),y_values.ravel()])

#绘制结果
predicted_labels = predicted_labels.reshape(x_values.shape)
plt.figure()
plt.clf()
plt.imshow(predicted_labels,interpolation = 'nearest',
	extent = (x_values.min(),x_values.max(),y_values.min(),y_values.max()),
	cmap = plt.cm.Paired,
	aspect = 'auto',origin = 'lower')

plt.scatter(data[:,0],data[:,1],marker = 'o',facecolors = 'none',edgecolors ='k',s = 30)

#叠加质心
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],marker = 'o',
	s = 200,linewidths = 3,color = 'k',zorder = 10,facecolors = 'black')

x_min,x_max = min(data[:,0])-1,max(data[:,0])+1
y_min,y_max = min(data[:,1])-1,max(data[:,1])+1

plt.title('Centoids and boundaries obtained using KMeans')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())
plt.show()
