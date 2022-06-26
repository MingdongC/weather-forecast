#————————————————————————————————————获取数据并归一化————————————————————————————————————————#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

#####训练集和测试集的路径
train_s5_aqi_path = '.\\dataset\\train_s5_aqi.csv'
train_s5_qx_path  = '.\\dataset\\train_s5_qx.csv'
test_s5_aqi_path = '.\\dataset\\test_s5_aqi.csv'
test_s5_qx_path  = '.\\dataset\\test_s5_qx.csv'

#####取训练数据和测试数据
train_s5_aqi = pd.read_csv(train_s5_aqi_path)
train_s5_qx = pd.read_csv(train_s5_qx_path)
test_s5_aqi = pd.read_csv(test_s5_aqi_path)
test_s5_qx = pd.read_csv(test_s5_qx_path)

###去除na值，替换成0
train_s5_aqi.fillna(0, inplace=True)
test_s5_aqi.fillna(0, inplace=True)

###将A1-A5站点的数据分开
train_s5_aqi_A1 = train_s5_aqi.loc[train_s5_aqi['siteid'] == 'A1', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]
train_s5_aqi_A2 = train_s5_aqi.loc[train_s5_aqi['siteid'] == 'A2', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]
train_s5_aqi_A3 = train_s5_aqi.loc[train_s5_aqi['siteid'] == 'A3', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]
train_s5_aqi_A4 = train_s5_aqi.loc[train_s5_aqi['siteid'] == 'A4', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]
train_s5_aqi_A5 = train_s5_aqi.loc[train_s5_aqi['siteid'] == 'A5', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]

test_s5_aqi_A1 = test_s5_aqi.loc[test_s5_aqi['siteid'] == 'A1', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]
test_s5_aqi_A2 = test_s5_aqi.loc[test_s5_aqi['siteid'] == 'A2', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]
test_s5_aqi_A3 = test_s5_aqi.loc[test_s5_aqi['siteid'] == 'A3', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]
test_s5_aqi_A4 = test_s5_aqi.loc[test_s5_aqi['siteid'] == 'A4', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]
test_s5_aqi_A5 = test_s5_aqi.loc[test_s5_aqi['siteid'] == 'A5', ['datetime', 'pm25', 'pm10', 'so2', 'o3', 'co', 'no2']]

train_s5_qx_A1 = train_s5_qx.loc[train_s5_qx['siteid'] == 'A1', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]
train_s5_qx_A2 = train_s5_qx.loc[train_s5_qx['siteid'] == 'A2', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]
train_s5_qx_A3 = train_s5_qx.loc[train_s5_qx['siteid'] == 'A3', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]
train_s5_qx_A4 = train_s5_qx.loc[train_s5_qx['siteid'] == 'A4', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]
train_s5_qx_A5 = train_s5_qx.loc[train_s5_qx['siteid'] == 'A5', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]

test_s5_qx_A1 = test_s5_qx.loc[test_s5_qx['siteid'] == 'A1', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]
test_s5_qx_A2 = test_s5_qx.loc[test_s5_qx['siteid'] == 'A2', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]
test_s5_qx_A3 = test_s5_qx.loc[test_s5_qx['siteid'] == 'A3', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]
test_s5_qx_A4 = test_s5_qx.loc[test_s5_qx['siteid'] == 'A4', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]
test_s5_qx_A5 = test_s5_qx.loc[test_s5_qx['siteid'] == 'A5', ['tem', 'vis', 'rhu', 'prs', 'pre', 'win_s', 'win_d']]


train_data_api_A1 = train_s5_aqi_A1.iloc[:, 1:7]
train_data_qx_A1 = train_s5_qx_A1.iloc[:, 2:9]

#####将数据标准差归一化处理
s = StandardScaler()
train_data_api_A1 = s.fit_transform(train_data_api_A1)
test_data_qx_A1 = s.fit_transform(train_data_qx_A1)

#####打印出来康康
plt.figure(figsize=(24,8))
plt.plot(train_data_qx_A1[:,:])
plt.plot(train_data_api_A1[:,:])
plt.legend(loc='upper left', fontsize= 24)
plt.show()

#####转成array
train_data_api = np.array(train_data_api_A1)
test_data_23 = np.array(train_data_qx_A1)
