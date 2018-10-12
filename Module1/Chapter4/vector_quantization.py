#week 5
'''
一个 k-means 聚类的主要应用是矢量量化
简单地说，矢量量化就是n维版本的四舍五入
但是和数字的四舍五入还是有细微差别
矢量量化可以用于图像压缩
用更少的位数储存每个像素实现压缩
'''
#python vector_quantization.py --input-file flower_image.jpg --num-bits 4
import numpy as np 
from scipy import misc
from sklearn import cluster
import matplotlib.pyplot as plt

import argparse 

#分析输入参数
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compress the input image \
            using clustering')
    parser.add_argument("--input-file", dest="input_file", required=True,
            help="Input image")
    parser.add_argument("--num-bits", dest="num_bits", required=False,
            type=int, help="Number of bits used to represent each pixel")
    return parser

#压缩图片函数
def compress_image(img,num_clusters):
	#将输入图片转化进（ num_samples , num_features )
	#运行 kmeans 算法数组
	X = img.reshape((-1,1))

	#在输入数据上运行kmeans
	kmeans = cluster.KMeans(n_clusters = num_clusters,n_init = 4,random_state = 5)
	kmeans.fit(X)
	centroids = kmeans.cluster_centers_.squeeze()
	labels = kmeans.labels_

	#分配每个值到最近的质心，重塑原始图片
	input_image_compressed = np.choose(labels,centroids).reshape(img.shape)
	return input_image_compressed

#输出图片效果
def plot_image(img,title):
	vmin = img.min()
	vmax = img.max()
	plt.figure()
	plt.title(title)
	plt.imshow(img,cmap = plt.cm.gray,vmin = vmin,vmax = vmax)

if __name__ == '__main__':
	args = build_arg_parser().parse_args()
	input_file = args.input_file
	num_bits = args.num_bits

	if not 1 <= num_bits <= 8:
		raise TypeError('Number of bits should be between 1 and 8')
	
	num_clusters = np.power(2,num_bits)

	#打印压缩率
	compression_rate = round(100*(8.0-args.num_bits)/8.0,2)
	print("\nThe size of the image will be reduced by a factor of",8.0/args.num_bits)
	print("\nCompression rate ="+str(compression_rate)+"%")

	#载入输入图片
	input_image = misc.imread(input_file,True).astype(np.uint8)

	#原始图片
	plot_image(input_image,'Original image')
	input_image_compressed = compress_image(input_image, num_clusters)
	plot_image(input_image_compressed,'Compressed image;compression rate='+str(compression_rate)+'%')
	plt.show()


