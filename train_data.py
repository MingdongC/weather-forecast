#--------------------------------------将训练数据进行预处理，得到batch_size、train_x和train_y--------------------------------------#
import numpy as np
from get_data import train_data_23

#####训练集预处理函数
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=len(train_data_23)):

    #每个batch的索引存储在batch_index里
    batch_index = []
    train_data = train_data_23[train_begin:train_end]

    '''
        这一句是用来标准化训练数据（用当前数据减去均值，再除以标准差）
        np.mean和np.std分别是求数据的均值和标准差，参数axis 缺省为求m*n的均值或者标准差\
        axis=0 表示求每一列的均值或标准差（压缩行），得到的shape为（1*n）\
        axis=1 表示求每一行的均值或标准差（压缩列），得到shape为（m*1） 
    '''
    #normalized_train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)

    normalized_train_data = train_data

    #定义train_x和train_y为list变量
    train_x, train_y = [], []

    #用一个loop将标准化后的train_data按每time_step为一个list放在train_x和train_y里,train_x和train_y为二维的list
    #建立batch_index
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            #这里不能直接用batch_index[i]索引，因为list未初始化
            batch_index.append(i)
        x = normalized_train_data[i:i+time_step, :4]
        y = normalized_train_data[i:i+time_step, 4, np.newaxis]        #np.newaxis是用来增加一个新的维度，放在行的位置就是增加行，列的位置是增加列
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return  batch_index, train_x, train_y

if __name__ == '__main__':

    get_train_data()