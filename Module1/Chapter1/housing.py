#week2
'''
AdaBoost算法 
两个核心步骤： 
1、权值调整：AdaBoost算法提高那些被前一轮基分类器错误分类样本的权值，而降低那些被正确分类样本的权值。从而使得那些没有得到正确分类的样本，由于权值的加大而受到后一轮基分类器的更大关注。 
2、基分类器组合：AdaBoost采用加权多数表决的方法： 
（1）加大分类误差率较小的弱分类器的权值，使得它在表决中起较大的作用。 
（2）减小分类误差率较大的弱分类器的权值，使得它在表决中起较小的作用。

scikit-learn基于AdaBoost算法提供了两个模型： 
AdaBoostClassifier用于分类问题 
AdaBoostRegressor用于回归问题

AdaBoostRegressor回归器 
ensemble.AdaBoostRegressor（）
参数
loss：一个字符串，指定了损失函数，可以为 
‘linear’：线性损失函数（默认） 
‘square’：平方损失函数 
‘exponential’：指数损失函数
方法
staged_predict(X)：返回一个数组，数组元素依次是每一轮迭代结束时尚未完成的集成回归器的预测值
staged_score(X,y[,sample_weight])：返回一个数字，数组元素依次是每一轮迭代结束时尚未完成的集成回归器的预测性能得分
---------------------

本文来自 LanboCSDN 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/lanbocsdn/article/details/78401095?utm_source=copy 
'''
'''
决策树的每个节点对输出产生影响
叶节点代表输出值，树枝代表中间的决策过程
AdaBoost代表自适应增强
它将不同算法得到的输出值进行加权求和，得到最终结果
前一阶段的信息将被反馈至后一阶段，难以分类的样本将在这一阶段被着重处理
AdaBoost对相同的数据集进行反复计算以达到满意的精度

这个任务的目标是建立房价和参数的回归模型
能够对未知参数的房价进行预估
'''
import numpy as np 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 

def plot_feature_importances(feature_importances,title,feature_names):
	#将重要程度的值规格化
	feature_importances = 100*(feature_importances/max(feature_importances))

	#对索引值降序排列
	index_sorted = np.flipud(np.argsort(feature_importances))

	#将标签的位置放在x轴
	pos = np.arange(index_sorted.shape[0])+0.5

	#绘制条形图
	plt.figure()
	plt.bar(pos,feature_importances[index_sorted],align = 'center')
	plt.xticks(pos,feature_names[index_sorted])
	plt.ylabel('Relative Importance')
	plt.title(title)
	plt.show()
	'''
	经过打印发现决策树下最重要因素是RM
	AdaBoost算法下最重要因素是LSTAT
	事实上最重要因素是LSTAT
	可见AdaBoost算法的优势
	'''

if __name__=='__main__':
	housing_data = datasets.load_boston()
	'''
	书上的下载链接挂掉了
	不过这个波士顿房价的数据库在sklearn内置了
	书上写的13个参数，网上查的是14个参数
	'''
	#将数据打乱
	X,y = shuffle(housing_data.data,housing_data.target,random_state = 7)

	#将数据二八分开用于测试和训练
	num_training = int(0.8*len(X))
	X_train,y_train = X[:num_training],y[:num_training]
	X_test,y_test = X[num_training:],y[num_training:]

	#决策树模型赋值
	dt_regressor = DecisionTreeRegressor(max_depth = 4)
	dt_regressor.fit(X_train,y_train)

	#Ada模型赋值
	ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4),n_estimators = 400,random_state = 7)
	ab_regressor.fit(X_train,y_train)

	y_pred_dt = dt_regressor.predict(X_test)
	mse = mean_squared_error(y_test,y_pred_dt)
	evs = explained_variance_score(y_test,y_pred_dt)
	print("\n#### Decision Tree performance ####")
	print("Mean squared explained_variance_scoreor = ",round(mse,2))
	print("Explained variance score = ",round(evs,2))

	y_pred_ab = ab_regressor.predict(X_test)
	mse = mean_squared_error(y_test,y_pred_ab)
	evs = explained_variance_score(y_test,y_pred_ab)
	print("\n#### AdaBoost performance ####")
	print("Mean squared error = ",round(mse,2))
	print("Explained variance score = ",round(evs,2))

	plot_feature_importances(dt_regressor.feature_importances_,'Decision Tree regressor',housing_data.feature_names)
	plot_feature_importances(ab_regressor.feature_importances_,"AdaBoost regressor",housing_data.feature_names)


