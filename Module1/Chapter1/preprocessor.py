#week1
import numpy as np
from sklearn import preprocessing

data = np.array([[3,-1.5,2,-5.4],
                 [0,4,-0.3,2.1],
                 [1,3.3,-1.9,-4.3]])

#mean removal均值移除
'''
变换后各维特征有0均值，单位方差，也叫零均值规范化
计算方式是将特征值减去均值，除以标准差
此处表示均值为0，标准偏离为1
'''
data_standardized = preprocessing.scale(data)
print("\nMean =",data_standardized.mean(axis=0))
print("Std deviation =",data_standardized.std(axis=0))


#min max scaling大小限定范围
'''
该方法进行的是线性变换
常用的变换到0-1之间
也可以是其他固定最小最大值的区间
此处将特征值限定在指定值0-1之间
'''
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(data)
print("\nMin max scaled data=",data_scaled)


#normalization规范化
'''
数据规范化是调整特征向量的值使之可以适应一个共有范围
机器学习中一个常用的规范特征向量值的方法是使它们和为1
重点在于使数据不会人为增加以保持向量的特征
'''
data_normalized = preprocessing.normalize(data,norm = 'l1')
print("\nL1 normalized data =",data_normalized)

#binarization二值化
'''
二值化能够将数据特征向量转化为布尔向量
当对数据有先前的知识时常用这种方式
'''
data_binarized = preprocessing.Binarizer(threshold = 1.4).transform(data)
print("\nBinarized data =",data_binarized)

#one hot encoding一位有效编码
'''
独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制
这种编码常用于机器学习中来标识特征
例如有红黄蓝三种颜色需要标识，则可以规定红[1,0,0],黄[0,1,0],蓝[0,0,1]

下面代码给出的是一个4维向量
其中第一维0，1，3三种特征值，用3位编码
第二维2，3两种特征值，用2位编码
第三维1，5，2，4四种特征值，用4位编码
第四维12，3两种特征值，用2位编码
共需要3+2+4+2=11位来编码
输出的是给出样本的独热码向量
'''
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0,2,1,12],
			[1,3,5,3],
			[2,3,2,12],
			[1,2,4,3]])
encoded_vector = encoder.transform([[2,3,5,3]]).toarray()
print("\nEncoded vector =",encoded_vector)