#week 7
import numpy as np
from functools import reduce
#map()函数接收两个参数，一个是函数，一个是序列，map将传入的函数依次作用到序列的每个元素，并把结果作为新的list返回。
#reduce把一个函数作用在一个序列[x1, x2, x3...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算
#在python3里面，map()的返回值是iterators,而不是list， 所以想要使用，需将iterator 转换成list 
def add3(input_array):
    return list(map(lambda x: x+3, input_array))

def mul2(input_array):
    return list(map(lambda x: x*2, input_array))

def sub5(input_array):
    return list(map(lambda x: x-5, input_array))

def function_composer(*args):
    return reduce(lambda f, g: lambda x: f(g(x)), args)

if __name__=='__main__':
    arr = np.array([2,5,4,7])

    print ("\nOperation: add3(mul2(sub5(arr)))")

    arr1 = add3(arr)
    arr2 = mul2(arr1)
    arr3 = sub5(arr2)
    print ("Output using the lengthy way:", arr3)

    func_composed = function_composer(sub5, mul2, add3)
    print ("Output using function composition:", func_composed(arr))

    print ("\nOperation: sub5(add3(mul2(sub5(mul2(arr)))))\nOutput:", \
        function_composer(mul2, sub5, mul2, add3, sub5)(arr))