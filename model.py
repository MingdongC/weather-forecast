#-----------------------------------------train_model的构建-----------------------------------------------------------#
import tensorflow as tf
from lstm_layer import lstm_layers
from train_data import get_train_data
from train_data import train_data_23

#####隐层数量
rnn_unit = 10
#input_size = 4  是因为需要输入的数据是minC,maxC,avgC,avgH 四列数据
input_size = 4
#output_size = 1 是因为预测输出为1个target
output_size = 1
#lr：learning_rate学习率
lr = 0.0006
epochs = 500


def lstm_model(batch_size=60, time_step=20, epochs=epochs, train_begin=0, train_end=len(train_data_23)):

    #placeholder是tf1.x里的占位符，shape[None：自动填补, time_step, input_size]
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm_layers(X)

    #tf.reduce_mean是对张量的某一个轴计算均值，若不指定，则是对所有轴计算均值，这里的loss是一个数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))

    #adaptive moment(Adam学习率自适应算法)算法利用一阶矩和二阶矩（样本平均值和样本平方的平均值）来优化loss函数
    train_optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    #Saver是tf里的一个保存训练好的模型的工具，max_to_keep是保留检查点最近的文件数量，默认为5
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    with tf.Session as sess:
        #在含有tf.Variable获知tf.get_Variable的环境下，需要用tf.global_variablies_initializer()进行初始化
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for step in range(len(batch_index-1)):
                #[train_op, loss]是将要执行的操作，feed_dict数据结构是字典，其元素是各种键值对
                #feed_dict: 将train_x和train_y的数据喂到X和Y里
                _, loss_ = sess.run([train_optimizer, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step+1]],
                                                                        Y: train_y[batch_index[step]:batch_index[step+1]]})
            if (i+1)%50 == 0:
                print("Number of epochs: ", i+1, "       loss: ", loss_)

        print("The train has finished")



