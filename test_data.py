#-------------------------------------将测试数据进行预处理,返回mean, std, test_x, test_y-----------------------------------#
import numpy as np
from get_data import train_data_23

data = train_data_23

#####对测试集数据进行预处理
def get_test_data(time_step=20, data=data, test_begin=0):
    test_data = data[test_begin:]
    mean = np.mean(test_data, axis=0)
    std = np.std(test_data, axis=0)
    normalized_test_data = test_data

    #"/"表示浮点数除法，“//”表示整数除法，返回整数；size 表示有size个sample
    size = (len(normalized_test_data) + time_step - 1) // time_step

    #将data_set按time_step等分，每一份是time_step,然后append进test_x里；test_x是一个二维的list；target数据用extend的方式装进test_y里
    test_x, test_y = [], []
    for i in range (size-1):
        x = normalized_test_data[ i*time_step : (i+1)*time_step, :4]
        y = normalized_test_data[ i*time_step : (i+1)*time_step, 4]
        test_x.append(x.tolist())
        test_y.extend(y.tolist())
    test_x.append((normalized_test_data[ (i+1)*time_step : , :4]).tolist())
    test_y.extend((normalized_test_data[ (i+1)*time_step : , 4]).tolist())
    return mean, std, test_x, test_y

