#week3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

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

	#X_train, X_test, y_train, y_test = cross_validation.train_test_split
	#建造简单贝叶斯回归
	classifier_gaussiannb = GaussianNB()
	classifier_gaussiannb.fit(X,y)
	y_pred = classifier_gaussiannb.predict(X)

	#计算分类精度
	accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]

	print("Accurary of the classifier = ",round(accuracy,2),"%")

	#绘制数据和边界
	#plot_classifier(classifier_gaussiannb,X,y)

	#分割训练和测试数据
	X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.25,random_state = 5)
	classifier_gaussiannb_new = GaussianNB()
	classifier_gaussiannb_new.fit(X_train,y_train)

	y_test_pred = classifier_gaussiannb_new.predict(X_test)

	accuracy = 100.0*(y_test == y_test_pred).sum()/X_test.shape[0]
	print("Accurary of the classifier = ",round(accuracy,2),"%")
	plot_classifier(classifier_gaussiannb_new,X_test,y_test)

	#用交叉验证评估精度
	'''
	过度拟合表示虽然在已有训练集上分类表型良好
	但在未知的数据上分类结果可能较差
	采用交叉验证的方法可以部分解决这个问题
	'''
	'''
	衡量机器学习的三个指标
	精度，召回率和f1分数
	精度是正确分类数据比全体数据
	召回率是正例中被分对的比例
	f1分数综合两者得出
	'''

	num_validations = 5
	accuracy =cross_validation.cross_val_score(classifier_gaussiannb,X,y,scoring = 'accuracy',cv = num_validations)
	print("Accurary :",str(round(100*accuracy.mean(),2)),"%")

	f1 = cross_validation.cross_val_score(classifier_gaussiannb,X,y,scoring = 'f1_weighted',cv = num_validations)
	print("F1 :",str(round(100*f1.mean(),2)),"%")

	precision = cross_validation.cross_val_score(classifier_gaussiannb,X,y,scoring = 'precision_weighted',cv = num_validations)
	print("Precision :",str(round(100*precision.mean(),2)),"%")

	recall = cross_validation.cross_val_score(classifier_gaussiannb,X,y,scoring = 'recall_weighted',cv = num_validations)
	print("Recall :",str(round(100*recall.mean(),2)),"%")

