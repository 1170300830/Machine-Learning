import numpy as np
import matplotlib.pyplot as plt

#创造一些样本数据
X = np.array([[3,1],[2,5],[1,8],[6,4],[5,2],[3,5],[4,7],[4,-1]])

#指派标签
y = [0,1,1,0,0,1,1,0]

#根据标签将数据划分成类
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

#分隔两种类的数据
line_x = range(10)
line_y = line_x

#绘制数据
plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],color = 'black',marker = 's')
plt.scatter(class_1[:,0],class_1[:,1],color = 'black',marker = 'x')
plt.plot(line_x,line_y,color = 'black',linewidth = 3)
plt.show()

'''
将y>x与y<x的分成两类
仅仅是一个简单的线性分类
'''




