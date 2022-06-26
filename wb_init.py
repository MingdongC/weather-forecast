#-----------------------------------------------权重和偏置初始化--------------------------------------------------#
import tensorflow as tf
from model import rnn_unit
from model import  input_size


# 输入层、输出层权重、偏置
weights = {

    #tf.random_normal用于从服从指定正太分布里取出shape的数量的个数，（shape, mean, stddev, dtype,seed, name）
    #这里的‘in’是存储初始化变量的字典键，shape是维数，mean是均值， stddev是标准差
    #输入是(4*10), 输出是（10*1）
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
    }
biases = {

    #常量初始化偏置值
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }

