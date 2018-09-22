#week2
import sys
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm

#导入数据
filename = sys.argv[1]
X = []
y = []
with open(filename,'r')as f:
	for line in f.readlines():
		data = [float(i) for i in line.split(',')]
		xt, yt = data[:-1], data[-1]
		X.append(xt)
		y.append(yt)

#训练数据和测试数据
num_training = int(0.8*len(X))
num_test = len(X)-num_training

X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])

X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

#线性回归
from sklearn import linear_model
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)

#多项式回归
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree = 10)
X_train_transformed = polynomial.fit_transform(X_train)

datapoint = [0.39,2.78,7.11]
#poly_datapoint = [polynomial.fit_transform(datapoint)]
poly_datapoint = [polynomial.fit_transform(np.array(datapoint).reshape(1, -1))]
'''
ValueError: Expected 2D array, got 1D array instead:
array=[0.39 2.78 7.11].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
这里是有关数组维度的错误，注意必须采用上述方式修改
很多在datapoint后面加.reshape的办法依旧会报错
list和array的差别要注意区别
'''

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
#print ("\nLinear regression:\n", linear_regressor.predict(datapoint))
print ("\nLinear regression:\n", linear_regressor.predict(np.array(datapoint).reshape(1, -1)))

#print ("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))
nsamples, nx, ny = np.array(poly_datapoint).shape
d2_poly_datapoint = np.array(poly_datapoint).reshape((nsamples,nx*ny))
print ("\nPolynomial regression:\n", poly_linear_model.predict(d2_poly_datapoint))
'''
ValueError: Found array with dim 3. Estimator expected <= 2.
需要将一个三维数组转换为二维
上述过程保持第一维度
并将其他两个维度展平
'''

'''
这个书中给的代码问题很多
网上搜索的改进方法很多也跑不出来
但最终还是用上面的代码跑出来了
但多项式回归值和书中的还是略有差异，尤其当degree = 10 时相差有0.05左右
可能是算法改进造成的吧
'''