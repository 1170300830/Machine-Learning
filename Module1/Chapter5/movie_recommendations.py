#week 8
'''
我们已经得到了电影推荐引擎的各个部分
余下的问题是进行拼装
'''
import json
import numpy as np

from euclidean_distance_score import euclidean_score
from pearson_correlation_score import pearson_score
from find_similar_users import find_similar_users

def generate_recommendations(dataset, user):
	#查看用户是否在数据集中
	if user not in dataset:
		raise TypeError('User ' + user + ' not present in the dataset')

	#计算该用户和数据集中其他用户的皮尔逊分数
	total_scores = {}
	similarity_sums = {}
	for u in [x for x in dataset if x != user]:
		similarity_score = pearson_score(dataset, user, u)

		if similarity_score <= 0:
			continue

		#寻找没有被用户评价过的电影
		for item in [x for x in dataset[u] if x not in dataset[user] or dataset[user][x] == 0]:
			total_scores.update({item: dataset[u][item] * similarity_score})
			similarity_sums.update({item: similarity_score})

	#如果用户观看过数据集中所有的电影，则没有什么电影可以推荐
	if len(total_scores) == 0:
		return ['No recommendations possible']

	#通过分数创建一个电影等级的规范化列表
	movie_ranks = np.array([[total/similarity_sums[item], item] 
		for item, total in total_scores.items()])

	#根据第一列元素降序排列
	movie_ranks = movie_ranks[np.argsort(movie_ranks[:, 0])[::-1]]

	#提取推荐电影
	recommendations = [movie for _, movie in movie_ranks]

	return recommendations
 
if __name__=='__main__':
	data_file = 'movie_ratings.json'

	with open(data_file, 'r') as f:
		data = json.loads(f.read())

	user = 'Michael Henry'
	print ("\nRecommendations for " + user + ":")
	movies = generate_recommendations(data, user) 
	for i, movie in enumerate(movies):
		print (str(i+1) + '. ' + movie)

	user = 'John Carson' 
	print ("\nRecommendations for " + user + ":")
	movies = generate_recommendations(data, user) 
	for i, movie in enumerate(movies):
		print (str(i+1) + '. ' + movie)