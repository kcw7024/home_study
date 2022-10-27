# import tensorflow as tf
# import numpy as np


# input_shape = (1, 1, 2, 2)
# x = np.arange(np.prod(input_shape))
# # x = np.arange(np.prod(input_shape)).reshape(input_shape)
# print(x)
# '''
# [[[[0 1]   
#    [2 3]]]]
# '''
# y = tf.keras.layers.ZeroPadding2D(padding=1)(x)
# print(y)
# '''
# tf.Tensor(
# [[[[0 0]  
#    [0 0]  
#    [0 0]  
#    [0 0]] 

#   [[0 0]  
#    [0 1]
#    [2 3]
#    [0 0]]

#   [[0 0]
#    [0 0]
#    [0 0]
#    [0 0]]]], shape=(1, 3, 4, 2), dtype=int32)

# '''
class A:
    def __init__(self):
        print('init')
    def __call__(self):
        print('call')
 

a = A()
a()

print(a)