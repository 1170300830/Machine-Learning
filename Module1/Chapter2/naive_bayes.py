#week3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

#定义绘图功能
def plot_classifier(classifier,X,y):
	#定义画图范围
	x_min,x_max = min(X[:,0])-1.0,max(X[:,0])+1.0
	y_min,y_max = min(X[:,1])-1.0,max(X[:,1])+1.0
	'''
	为了清晰将边界界定在最大最小值上下1.0
	'''

	#定义网格网的步长
	step_size = 0.01

	#定义网格网
	x_values,y_values = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))

	#计算分类输出
	mesh_output = classifier.predict(np.c_[x_values.ravel(),y_values.ravel()])

	#重塑数组
	mesh_output = mesh_output.reshape(x_values.shape)

	#彩色绘制输出结果
	plt.figure()

	#选择一个颜色方案
	plt.pcolormesh(x_values,y_values,mesh_output,cmap = plt.cm.gray)
	'''
	用3d绘图器绘制2d点
	用一个颜色方案将关联点绘制在不同颜色区域
	'''

	plt.scatter(X[:,0],X[:,1],c = y,s = 80,edgecolors = 'black',linewidth = 1,cmap = plt.cm.Paired)

	#明确图形边界
	plt.xlim(x_values.min(),x_values.max())
	plt.ylim(y_values.min(),y_values.max())

	#指定x和y轴刻度,给轴标定数字
	plt.xticks((np.arange(int(min(X[:,0])-1),int(max(X[:,0])+1),1.0)))
	plt.yticks((np.arange(int(min(X[:,1])-1),int(max(X[:,1])+1),1.0)))

	plt.show()

if __name__ == '__main__':
	#加载数据
	input_file = 'data_multivar.txt'

	X = []
	y = []
	with open(input_file,'r')as f:
		for line in f.readlines():
			data = [float(x) for x in line.split(',')]
			X.append(data[:-1])
			y.append(data[-1])

	X = np.array(X)
	y = np.array(y)
	#建造简单贝叶斯回归
	classifier_gaussiannb = GaussianNB()
	classifier_gaussiannb.fit(X,y)
	y_pred = classifier_gaussiannb.predict(X)

	#计算分类精度
	accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]

	print("Accurary of the classifier = ",round(accuracy,2),"%")

	#绘制数据和边界
	plot_classifier(classifier_gaussiannb,X,y)