#week4
'''
预测建模是数据分析最活跃的领域
它常见地用于数据挖掘以预测未来趋势
之前章节我们是用已知响应的数据训练模型
并用一些数据指标评价模型
再用模型进行预测
我们尝试了很多算法来建立预测模型
这一章中我们将用支持向量机来建造线性和非线性模型
进行预测之前要寻找影响系统的特征向量
'''

'''
支持向量机是可监督学习模型中用于建造分类器和回归器的
一个支持向量机通过求解一系列的数学等式可以发现分割两组点的最佳边界
'''

import numpy as np 
import matplotlib.pyplot as plt 
import utilities

#加载输入数据
input_file = 'data_multivar.txt'
X,y = utilities.load_data(input_file)

#分割数据为不同的类
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None', edgecolors='black', marker='s')
plt.title('Input data')

'''
在图片的结果中，包含实心正方形和空心正方形
在机器学习行话中，我们说数据包含两个类
我们的目的是将实心正方形从空心正方形里分离出来
'''

#分割数据集为训练和测试数据
from sklearn import cross_validation
from sklearn.svm import SVC

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.25,random_state = 5)

#用线性内核初始化
params = {'kernel':'linear'}
classifier = SVC(**params)

#训练线性SVM分类器
classifier.fit(X_train,y_train)

#查看分类器效果
utilities.plot_classifier(classifier,X_train,y_train,'Training dataset')
'''
书上没给utilities.plot_classifier的写法
我直接粘贴了课程附带里的，没有自己写
我详细阅读给出的代码，大体与前一章节绘图方式一致
'''

#测试在测试集上的效果
y_test_pred = classifier.predict(X_test)
utilities.plot_classifier(classifier,X_test,y_test,'Test dataset')
plt.show()

#计算训练集的精度
from sklearn.metrics import classification_report
target_names = ['Class'+str(int(i)) for i in set(y)]
print("\n"+"#"*30)
'''
还有这种操作？
'''
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train,classifier.predict(X_train),target_names = target_names))
print("#"*30+"\n")

print ("#"*30)
print ("\nClassification report on test dataset\n")
print (classification_report(y_test, y_test_pred, target_names=target_names))
print ("#"*30 + "\n")