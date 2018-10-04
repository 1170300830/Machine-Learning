#week4
'''
预测建模是数据分析最活跃的领域
它常见地用于数据挖掘以预测未来趋势
之前章节我们是用已知响应的数据训练模型
并用一些数据指标评价模型
再用模型进行预测
我们尝试了很多算法来建立预测模型
这一章中我们将用支持向量机来建造线性和非线性模型
进行预测之前要寻找影响系统的特征向量
'''

'''
支持向量机是可监督学习模型中用于建造分类器和回归器的
一个支持向量机通过求解一系列的数学等式可以发现分割两组点的最佳边界
'''

import numpy as np 
import matplotlib.pyplot as plt 
import utilities

#加载输入数据
input_file = 'data_multivar.txt'
X,y = utilities.load_data(input_file)

#分割数据为不同的类
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()