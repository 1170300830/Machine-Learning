#week2
'''
bagging,random forest,AdaBoost算法的理解
三者都是基于决策树的集成算法
bagging是每个决策树抽取一定数量样本后进行分析，通过每个决策树分析结果“投票”决定最终结果
random forest也是每个决策树抽取一定数量样本，但每个决策树会随机选择一些特征进行分析，最终加权决定最终结果
random forest改进在于避免一些特征对最终结果产生太大影响，从而更加精确，不过模型的可解释性有所下降
AdaBoost通过多轮分析产生最终结果，后一轮将对前一轮拟合效果不好的样本着重关注，加大其权值，最后寻求一个更高的精度
'''

import csv
from sklearn.ensemble import RandomForestRegressor
from housing import plot_feature_importances
import sys
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, explained_variance_score

def load_dataset(filename):
	#file_reader = csv.reader(open(filename,'rb'),delimiter = ',')
	file_reader = csv.reader(open(filename,'r'),delimiter = ',')
	X,y = [],[]
	for row in file_reader:
		X.append(row[2:14])
		'''
		row[2:13]时Mean squared error =  357096.36，Explained variance score =  0.89
		row[2:15]时Mean squared error =  21896.2，Explained variance score =  0.99
		原因是当取到后两个特征时，输出就是两者简单相加，拟合最佳
		同时图形能清晰反映这一点
		'''

		y.append(row[-1])

	#提取特征名字
	feature_names = np.array(X[0])

	#移去作为特征名字的第一行
	return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

if __name__=='__main__':
	X,y,feature_names = load_dataset(sys.argv[1])
	#打乱顺序
	X,y = shuffle(X,y,random_state = 7)
	#划分训练和测试数据
	num_training = int(0.9*len(X))
	X_train,y_train = X[:num_training],y[:num_training]
	X_test,y_test = X[num_training:],y[num_training:]
	#训练回归模型
	rf_regressor = RandomForestRegressor(n_estimators = 1000,max_depth = 10,min_samples_split = 1)
	#此处因为报错修改了sklearn.ensemble下tree函数的源代码
	rf_regressor.fit(X_train,y_train)
	'''
	n_estimators表示森林中决策树的数量,max_depth表示决策树的最大深度，min_samples_split表示需要在一棵树上被节点分割的样本点数量 
	'''

	#评估随机森林的表现
	y_pred = rf_regressor.predict(X_test)
	mse = mean_squared_error(y_test,y_pred)
	evs = explained_variance_score(y_test,y_pred)
	print("\n#### Random Forst regressor performance ####")
	print("Mean squared error = ",round(mse,2))
	print("Explained variance score = ",round(evs,2))

	#绘图
	plot_feature_importances(rf_regressor.feature_importances_,'Random Forest regressor',feature_names)







