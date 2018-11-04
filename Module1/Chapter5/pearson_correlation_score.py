#week 8

'''
欧几里得距离分数有短板
引入频繁在推荐引擎中使用的皮尔逊相关分数
'''

import json
import numpy as np

#确认使用者在数据集中
def pearson_score(dataset, user1, user2):
	if user1 not in dataset:
		raise TypeError('User ' + user1 + ' not present in the dataset')

	if user2 not in dataset:
		raise TypeError('User ' + user2 + ' not present in the dataset')

	#得到共有的电影比率
	rated_by_both = {}

	for item in dataset[user1]:
		if item in dataset[user2]:
			rated_by_both[item] = 1

	num_ratings = len(rated_by_both)

	#如果没有共同的电影，返回0
	if num_ratings == 0:
		return 0

	#计算所有共同喜好比率的和
	user1_sum = np.sum([dataset[user1][item] for item in rated_by_both])
	user2_sum = np.sum([dataset[user2][item] for item in rated_by_both])

	#计算所有共同喜好比率平方和
	user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in rated_by_both])
	user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in rated_by_both])

	#计算共同比率的产品和
	product_sum = np.sum([dataset[user1][item] * dataset[user2][item] for item in rated_by_both])

	#计算皮尔逊相关系数
	Sxy = product_sum - (user1_sum * user2_sum / num_ratings)
	Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
	Syy = user2_squared_sum - np.square(user2_sum) / num_ratings
    
	if Sxx * Syy == 0:
		return 0

	return Sxy / np.sqrt(Sxx * Syy)

if __name__=='__main__':
	data_file = 'movie_ratings.json'

	with open(data_file, 'r') as f:
		data = json.loads(f.read())

	user1 = 'John Carson'
	user2 = 'Michelle Peterson'

	print ("\nPearson score:")
	print (pearson_score(data, user1, user2)) 


