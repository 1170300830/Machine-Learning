#week4
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation

#从输入文件中载入数据
def load_data(input_file):
	X = []
	y = []
	with open(input_file,'r')as f:
		for line in f.readlines():
			data = [float(x) for x in line.split(',')]
			X.append(data[:-1])
			y.append(data[-1])

	X = np.array(X)
	y = np.array(y)
	return X,y

