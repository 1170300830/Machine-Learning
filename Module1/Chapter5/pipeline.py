#week 7
'''
建造机器学习流水线
流水线功能包括预处理，特征选择，监督学习和无监督学习等等
这个流水线将输出特征向量，选择 top k特征并用随机森林分类器分类
'''

from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

#得到一些样本点
X, y = samples_generator.make_classification(
	n_informative=4, n_features=20, n_redundant=0, random_state=5)
'''
得到了一个20维的特征向量
'''

#挑选k个最佳特征
selector_k_best = SelectKBest(f_regression, k=10)

#随机森林分类器
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)

#建造机器学习流水线
pipeline_classifier = Pipeline([('selector', selector_k_best), ('rf', classifier)])
'''
在建造机器学习流水线时可以自由的给其中的元素分配名字
'''

'''
我们也可以变更流水线中区块的参数
例如我们假如想修改 selector_k_best 和 classifier的参数
可以用下面的代码
'''
pipeline_classifier.set_params(selector__k=6,
	rf__n_estimators=25)

#训练分类器
pipeline_classifier.fit(X, y)

#预测输出
prediction = pipeline_classifier.predict(X)
print ("\nPredictions:\n", prediction)

#估计分类器的表现
print ("\nScore:", pipeline_classifier.score(X, y))  

#输出被挑选的特征
features_status = pipeline_classifier.named_steps['selector'].get_support()
selected_features = []
for count, item in enumerate(features_status):
	if item:
		selected_features.append(count)

print ("\nSelected features (0-indexed):", ', '.join([str(x) for x in selected_features]))
