#w1
from sklearn import preprocessing
#这个包里包含许多数据预处理功能

label_encoder = preprocessing.LabelEncoder()
#这个对象知道如何理解词汇标签

input_classes = ['audi','ford','audi','toyota','ford','bmw']

#将四种不同的词汇标签转换为数字编码
label_encoder.fit(input_classes)
print("\nClass mapping:")
for i,item in enumerate(label_encoder.classes_):
	print(item,'-->',i)

#词汇标签转换为数字编码之后，用两种方式分别打印一个已知的标签数组
#前一段代码是将词汇编码转换为数字编码，后一段则恰好相反
labels = ['toyota','ford','audi']
encoded_labels = label_encoder.transform(labels)
print("\nLabels = ",labels)
print("Encoded labels = ",list(encoded_labels))

encoded_labels = [2,1,0,3,1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("\nEncoded labels = ",encoded_labels)
print("Decoded labels = ",list(decoded_labels))