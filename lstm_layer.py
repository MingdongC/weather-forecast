#----------------------------------------------------LSTM层定义-------------------------------------------------------#
import tensorflow as tf
from wb_init import weights
from wb_init import biases
from model import input_size
from model import rnn_unit

#####lstm层定义
def lstm_layers(X):

    #输入X的大小是一个batch的大小，所以取它的行数就是batch_size
    batch_size = tf.shape(X)[0]

    #time_step是记忆周期，表明当前数据与前time_step个数据有关系
    time_step = tf.shape(X)[1]

    w_in = weights['in']
    b_in = biases['in']

    # tf.reshape的参数（tensor, shape, name=None）,需要将X（tensor变量）从3维降成两维进行线性计算，计算后的结果作为隐藏层的输入
    input = tf.reshape(X,[-1, input_size])
    #输入先进行一次线性作为隐藏层的输入，公式是 input_rnn = input * w_in + b_in
    input_rnn = tf.matmul(input,w_in) + b_in

    #转回没计算之前X的shape
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])

    # cell为letm前向传播的另一条带Ct和ht是并行的。cell用来忘记部分以前的记忆，并选择性的增加当下的记忆来更新自身和输出
    # 参数：num_units,forget_bias,state_is_tuple,activation,reuse,name,dtype
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # dynamic_rnn是tensorflow封装的用来实现rnn的函数，参数：cell,inputs,sequence_length,initial_state,dtype,parallel_iterations.swap_memory,time_major,scope
    #output_rnn：当层的ht ； final_states：当层的Ct
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)

    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']

    #输出层
    pred = tf.matmul(output, w_out) + b_out

    return pred, final_states
