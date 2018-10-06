#week 4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import utilities 
'''
#删去#这个代码因为年代久远的问题有很多问题
#删去#比如注释掉的代码部分无论如何修改都跑不出来
原因是首先按照课程所给代码会报错
ValueError: Expected 2D array, got 1D array instead
当使用reshape修改后又会报错
ValueError: X.shape[1] = 1 should be equal to 2, the number of features at training time
前者要求是一个二维数组，后者要求有两个值在里面...

机智如我在写这个的时候发现的问题的解决方法
不说了，看代码，都是泪...
'''
#加载数据
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

#训练边界划分
from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

params = {'kernel': 'rbf'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)


#测量到边界值的距离
input_datapoints = np.array([[2, 1.5], [8, 9], [4.8, 5.2], [4, 4], [2.5, 7], [7.6, 2], [5.4, 5.9]])

print("\nDistance from the boundary:")
for i in input_datapoints:
	i = [i]
	#万恶之源
	print (i, '-->', classifier.decision_function(i)[0])

#信度测量
params = {'kernel': 'rbf', 'probability': True}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

print ("\nConfidence measure:")
for i in input_datapoints:
	i = [i]
	#万恶之源
	print (i, '-->', classifier.predict_proba(i)[0])


utilities.plot_classifier(classifier, input_datapoints, [0]*len(input_datapoints), 'Input datapoints', 'True')
plt.show()

'''
距离边界的信息给了很多关于数据点的信息
但是我们不能直接得到分类器的信度
这里就要直接去测量信度
这里使用的数学工具是普拉特缩放
它将距离指标转化为类之间的概率指标
'''