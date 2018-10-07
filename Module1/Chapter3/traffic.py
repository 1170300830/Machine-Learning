#week4
'''
这个项目是将 SVM 做一个回归器分析交通
数据集是洛杉矶道奇队棒球主场时通过的汽车数量
特征包括日期，时间，对手球队，是否将有一场比赛，汽车通过数量
'''

#估计交通的SVM回归器
import numpy as np 
from sklearn import preprocessing
from sklearn.svm import SVR

input_file = 'traffic_data.txt'
'''
数据集太大，调试程序很麻烦
用traffic_data_test.txt调试程序
'''

#读取数据
X = []
count = 0
with open(input_file,'r')as f:
	for line in f.readlines():
		data = line[:-1].split(',')
		X.append(data)

X = np.array(X)

#将字符串数据转换为数字数据
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
	if item.isdigit():
		X_encoded[:,i] = X[:,i]
	else:
		label_encoder.append(preprocessing.LabelEncoder())
		X_encoded[:,i] = label_encoder[-1].fit_transform(X[:,i])

X = X_encoded[:,:-1].astype(int)
y = X_encoded[:,-1].astype(int)

#用径向基函数建立 SVM 回归器
#建立 SVM
params = {'kernel':'rbf',
			'C':10.0,
			'epsilon':0.2}
regressor = SVR(**params)
regressor.fit(X,y)
'''
上面的式子C说明误判的惩罚
epsilon 明确不适用惩罚的限制
'''

#交叉验证
import sklearn.metrics as sm 
y_pred = regressor.predict(X)
print("Mean absolute error =",round(sm.mean_absolute_error(y,y_pred),2))

#用单个数据样例测试编码
input_data = ['Tuesday','13:35','San Francisco','yes']
input_data_encoded = [-1]*len(input_data)
count = 0
for i,item in enumerate(input_data):
	if item.isdigit():
		input_data_encoded[i] = int(input_data[i])
	else:
		#input_data_encoded[i] = int(label_encoder[count].transform(input_data[i]))
		input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
		count = count + 1

input_data_encoded = np.array(input_data_encoded)

#预测和打印特殊数据点的输出
print("Predicted traffic:",int(regressor.predict([input_data_encoded])[0]))
