#week4
'''
提取 SVM 的最佳超级参数
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, grid_search, cross_validation
from sklearn.metrics import classification_report

import utilities 

input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

#设置交叉验证的超级参数
parameter_grid =  [{'kernel':['linear'],'C':[1,10,50,600]},
					{'kernel':['poly'],'degree':[2,3]},
					{'kernel':['rbf'],'gamma':[0.01,0.001],'C':[1,10,50,600]},
					]

#定义我们希望使用的指标
metrics = ['precision','recall_weighted']

#开始寻找对于每个指标的最佳超级参数
for metric in metrics:
	print("\n#### Searching optimal hyperparameters for",metric)

	classifier = grid_search.GridSearchCV(svm.SVC(C=1),
		parameter_grid, cv = 5,scoring = metric)
	classifier.fit(X_train,y_train)

	#查看分数
	print("\nScore across the parameter grid:")
	for params,avg_score,_ in classifier.grid_scores_:
		print(params,'-->',round(avg_score,3))

	#查看最佳超级参数设置
	print("\nHightest scoring parameter set:",classifier.best_params_)

	y_true,y_pred = y_test,classifier.predict(X_test)
	print("\nFull performance report:\n")
	print(classification_report(y_true,y_pred))




