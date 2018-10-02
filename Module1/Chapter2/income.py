#week3
'''
预测个人的收入是大于五万还是小于五万
数字和字符串混合的数据集合
数字数据不可以使用标签编码
这个系统需要能同时处理数字数据和非数字数据
'''

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB 

input_file = 'adult.data.txt'

#读取数据
X = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_images_threshold = 10000
'''
当某一个数据集的样本选取过多时
分类器将会偏向数据集
所以应当等量的选取每个数据集中的数据
'''

with open(input_file,'r')as f:
	for line in f.readlines():
		if '?' in line:
			continue
		data = line[:-1].split(', ')

		if data[-1] == '<=50K' and count_lessthan50k < num_images_threshold:
			X.append(data)
			count_lessthan50k = count_lessthan50k + 1

		elif data[-1] == '>50K' and count_morethan50k < num_images_threshold:
			X.append(data)
			count_morethan50k = count_morethan50k + 1

		if count_lessthan50k >= num_images_threshold and count_morethan50k>=num_images_threshold:
			break

X = np.array(X)

#将字符串数据转化为数字数据
label_encoder = [] 
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
	if item.isdigit():
		X_encoded[:, i] = X[:, i]
	else:
		label_encoder.append(preprocessing.LabelEncoder())
		X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

#创建分类器
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X,y)

#交叉验证
from sklearn import cross_validation
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.25,random_state = 5)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb.predict(X_test)

#计算分类器的f1分数
f1 = cross_validation.cross_val_score(classifier_gaussiannb,X,y,scoring = 'f1_weighted',cv = 5)
print("F1 score: ",str(round(100*f1.mean(),2)),"%")

#测试分类器在单个数据的效果 input_data = ['39', 'State-gov', '77516', 'Bachelors', '13',
input_data = ['39', 'State-gov', '77516', 'Bachelors', '13', 'Never-married', \
	'Adm-clerical', 'Not-in-family', 'White', 'Male', '2174', '0', '40', 'United-States'] 
count = 0

input_data_encoded = [-1]*len(input_data)
for i,item in enumerate(input_data):
	if item.isdigit():
		input_data_encoded[i] = int(input_data[i])
	else:
		#input_data_encoded[i] = int(label_encoder[count].transform(input_data[i]))
		input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))

		count = count + 1

input_data_encoded = np.array(input_data_encoded)

#预测和打印一个数据点的输出
#output_class = classifier_gaussiannb.predict(input_data_encoded)
output_class = classifier_gaussiannb.predict([input_data_encoded])
print(label_encoder[-1].inverse_transform(output_class)[0])


					        				