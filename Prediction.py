#-------------------------------------------------预测模型-------------------------------------------------------------#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from test_data import get_test_data
from lstm_layer import lstm_layers
from model import  input_size

def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step,test_begin=0)
    #tf.variable_scope创建一个名为“sec_lstm”的变量空间，pred,_ 是里面的变量
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm_layers(X)

    #保存全局变量
    saver=tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[4]+mean[4]
        test_predict=np.array(test_predict)*std[4]+mean[4]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)]))  #mean absolute error
        print("The MAE of this predict:",acc)
        #以折线图表示结果
        plt.figure(figsize=(24,8))
        plt.plot(list(range(len(test_predict))), test_predict, color='b',label = 'prediction')
        plt.plot(list(range(len(test_y))), test_y,  color='r',label = 'origin')
        plt.legend(fontsize=24)
        plt.show()