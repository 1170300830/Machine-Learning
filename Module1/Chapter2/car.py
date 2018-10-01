#week 3
'''
汽车的参数为：
购买价格，维护价格，门数，人数，防护罩和安全性
利用随机森林建造分类器
'''
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

input_file = 'car.data.txt'

#读取数据
X = []
y = []
count =0
with open(input_file,'r')as f:
	for line in f.readlines():
		data = line[:-1].split(',')
		X.append(data)
X = np.array(X)

#使用标签编码将字符数据转换为数字数据
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
	label_encoder.append(preprocessing.LabelEncoder())
	X_encoded[:,i] = label_encoder[-1].fit_transform(X[:,i])

X = X_encoded[:,:-1].astype(int)
y = X_encoded[:,-1].astype(int)

#建造随机森林分类器
params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)

#交叉验证
from sklearn import cross_validation
accuracy = cross_validation.cross_val_score(classifier,X,y,scoring = 'accuracy',cv = 3)
print("Accuracy of the classifier:",str(round(100*accuracy.mean(),2)),"%")

#利用实例测试程序
input_data = ['vhigh','vhigh','2','2','small','low']
input_data_encoded = [-1]*len(input_data)
for i,item in enumerate(input_data):
    #input_data_encoded[i] = int(label_encoder[i].transform(input_data[i]))
    input_data_encoded[i] = int(label_encoder[i].transform([input_data[i]]))

input_data_encoded = np.array(input_data_encoded)

#预测这个数据点的输出类
#output_class = classifier.predict(input_data_encoded)
output_class = classifier.predict([input_data_encoded])
print("Output class:",label_encoder[-1].inverse_transform(output_class)[0])

'''
超级参数影响分类效果
n_estimators 和 max_depth
验证曲线帮助我们了解超级参数如何影响训练分数
'''
#验证曲线
from sklearn.learning_curve import validation_curve
classifier = RandomForestClassifier(max_depth = 4, random_state = 7)
parameter_grid = np.linspace(25,200,8).astype(int)
train_scores,validation_scores = validation_curve(classifier,X,y,"n_estimators",parameter_grid,cv = 5)
print ("\n##### VALIDATION CURVES #####")
print ("\nParam: n_estimators\nTraining scores:\n", train_scores)
print ("\nParam: n_estimators\nValidation scores:\n", validation_scores)
'''
固定 max_depth
验证估计者数量和精度的关系
估计者数量从25到200，从中取8个点进行绘图
'''

#绘制曲线
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

classifier = RandomForestClassifier(n_estimators=20, random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int)
train_scores, valid_scores = validation_curve(classifier, X, y, "max_depth", parameter_grid, cv=5)
print ("\nParam: max_depth\nTraining scores:\n", train_scores)
print ("\nParam: max_depth\nValidation scores:\n", validation_scores)
'''
固定 n_estimators
验证估计者最大深度和精度的关系
'''

#绘制曲线
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Validation curve')
plt.xlabel('Maximum depth of the tree')
plt.ylabel('Accuracy')
plt.show()

'''
学习曲线帮助我们了解训练集的大小如何影响机器学习模型
这对解决计算约束的问题很重要
'''
#学习曲线
from sklearn.learning_curve import learning_curve
classifier = RandomForestClassifier(random_state = 7)
parameter_grid = np.array([200,500,800,1000])
train_sizes,train_scores,validation_scores = learning_curve(classifier,X,y,train_sizes = parameter_grid,cv = 5)

print ("\n##### LEARNING CURVES #####")
print ("\nTraining scores:\n", train_scores)
print ("\nValidation scores:\n", validation_scores)

plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()

