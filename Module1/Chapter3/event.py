#week4
'''
建立一个 SVM 来预测进出一栋建筑物的人数
特征有星期、日期、时间、出建筑物人数、进建筑物人数、输出是否指向事件
前五个是输入数据，任务是预测是否在建筑内有事件发生

后一个数据文件是进阶版的
它与前一个数据文件不同在于准确描述了事件类型
'''

import numpy as np 
from sklearn import preprocessing
from sklearn.svm import SVC

#input_file = 'building_event_binary.txt'
input_file = 'building_event_multiclass.txt'

#读取数据
X = []
count =0
with open(input_file,'r')as f:
	for line in f.readlines():
		data = line[:-1].split(',')
		X.append([data[0]]+data[2:])

X = np.array(X)

#将数据转化为数学形式
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

#用径向基函数，普特拉缩放和类平衡来训练SVM
params = {'kernel':'rbf',
			'probability':True,
			'class_weight':'balanced'}
classifier = SVC(**params)
classifier.fit(X,y)

#交叉验证
from sklearn import cross_validation
accuracy = cross_validation.cross_val_score(classifier,X,y,scoring = 'accuracy',cv = 3)
print("Accurary of the classifier:",str(round(100*accuracy.mean(),2)),'%')

#测试单独的数据例子
input_data = ['Tuesday','12:30:00','21','23']
input_data_encoded = [-1]*len(input_data)
count = 0
for i,item in enumerate(input_data):
	if item.isdigit():
		input_data_encoded[i] = int(input_data[i])
	else:
		input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
		count = count + 1

input_data_encoded = np.array(input_data_encoded)

#预测和打印一个特殊点的输出
output_class = classifier.predict([input_data_encoded])
print("Output class:",label_encoder[-1].inverse_transform(output_class)[0])