#week 8

'''
前面已经为机器学习流水线和最近领域分类做了充足的准备
下面开始讨论推荐引擎
为了构建推荐引擎
需要定义相似度指标数据库中和目标用户相似的用户
我们将聚焦于电影推荐引擎
首先看看如何计算两个用户间的欧几里得分数
'''
import json
import numpy as np

def euclidean_score(dataset, user1, user2):
	#首先检查两个用户现在是否在数据集中
	if user1 not in dataset:
		raise TypeError('User ' + user1 + ' not present in the dataset')
	if user2 not in dataset:
		raise TypeError('User ' + user2 + ' not present in the dataset')

	#提取两个用户共有的电影比率
	rated_by_both = {} 

	for item in dataset[user1]:
		if item in dataset[user2]:
			rated_by_both[item] = 1

	#如果没有共有电影，则相似度为0
	if len(rated_by_both) == 0:
		return 0

	#计算方差和的平方根，规范化到零到一上得到分数
	squared_differences = [] 

	for item in dataset[user1]:
		if item in dataset[user2]:
			squared_differences.append(np.square(dataset[user1][item] - dataset[user2][item]))
        
	return 1 / (1 + np.sqrt(np.sum(squared_differences)))
	'''
	如果比率相似，方差和将会非常小
	分数将会非常高
	'''

if __name__=='__main__':
	data_file = 'movie_ratings.json'

	with open(data_file, 'r') as f:
		data = json.loads(f.read())

	user1 = 'John Carson'
	user2 = 'Michelle Peterson'
	print ("\nEuclidean score:")
	print (euclidean_score(data, user1, user2)) 	
