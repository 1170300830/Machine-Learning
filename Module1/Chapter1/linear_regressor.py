import sys
import numpy as np

#导入数据
filename = sys.argv[1]
X = []
y = []
with open(filename,'r')as f:
	for line in f.readlines():
		xt,yt = [float(i) for i in line.split(',')]
		X.append(xt)
		y.append(yt)

#训练数据和测试数据
num_training = int(0.8*len(X))
num_test = len(X)-num_training

X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])
'''
上述代码使用了八成数据进行训练，两成数据进行测试
'''

#创建一个线性回归对象
from sklearn import linear_model
linear_regressor = linear_model.LinearRegression()

#使用训练集训练模型
linear_regressor.fit(X_train,y_train)

#观察训练结果
import matplotlib.pyplot as plt 
y_train_pred = linear_regressor.predict(X_train)

plt.figure()
plt.scatter(X_train,y_train,color = 'green')
plt.plot(X_train,y_train_pred,color = 'black',linewidth = 4)
plt.title('Training data')
plt.show()
'''
上述代码仅仅是利用已知数据训练了模型
但对于未知数据是否匹配模型还不可知
所以接下来要用余下的两成数据测试模型
'''

y_test_pred = linear_regressor.predict(X_test)
plt.scatter(X_test,y_test,color = 'green')
plt.plot(X_test,y_test_pred,color = 'black',linewidth = 4)
plt.title('Test data')
plt.show()

#测试精度
import sklearn.metrics as sm

print("Mean absolute error = ",round(sm.mean_absolute_error(y_test,y_test_pred),2))
'''
平均绝对误差
'''
print("Mean squared error = ",round(sm.mean_squared_error(y_test,y_test_pred),2))
'''
平均方差
'''
print("Median absolute error = ",round(sm.median_absolute_error(y_test,y_test_pred),2))
'''
中间绝对误差
'''
print("Explain variance score = ",round(sm.explained_variance_score(y_test,y_test_pred),2))
'''
解释方差分数
'''
print("R2 score = ",round(sm.r2_score(y_test,y_test_pred),2))
'''
R2分数
'''
'''
一般为了避免繁琐
仅仅挑选一到两个指标测试精度
常用绝对方差（越低越好）和解释方差分数（越高越好）
'''