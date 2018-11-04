#week 8

'''
推荐引擎的一个关键任务就是寻找相似用户
这个项目将帮助发现数据集中的相似用户
'''

import json
import numpy as np
from pearson_correlation_score import pearson_score

'''
发现输入用户的相似用户
一共需要三个参数，数据集，输入用户，期待相似用户个数
'''
def find_similar_users(dataset, user, num_users):
	#检查用户是否在数据集中
	if user not in dataset:
		raise TypeError('User ' + user + ' not present in the dataset')

	#计算与所有用户的皮尔逊分数
	scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if user != x])

	#用第二列来排序分数
	scores_sorted = np.argsort(scores[:, 1])

	#降序排列分数
	scored_sorted_dec = scores_sorted[::-1]

	#提取前排指标
	top_k = scored_sorted_dec[0:num_users] 

	return scores[top_k] 

if __name__=='__main__':
	data_file = 'movie_ratings.json'

	with open(data_file, 'r') as f:
		data = json.loads(f.read())

	user = 'John Carson'
	print ("\nUsers similar to " + user + ":\n")
	similar_users = find_similar_users(data, user, 3) 
	print ("User\t\t\tSimilarity score\n")
	for item in similar_users:
		print (item[0], '\t\t', round(float(item[1]), 2))