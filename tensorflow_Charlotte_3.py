import numpy as np
import tensorflow as tf
import sys

#使用sklearn中加洲房價數據
from sklearn.datasets import fetch_california_housing
#ScandardScaler: 將特徵進行標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#取得數據
housing = fetch_california_housing()
m, n = housing.data.shape
print(housing.keys())
print(housing.feature_names)#房價特徵: 房齡、經度緯度等
print(housing.target)
print(housing.DESCR)
print(housing.data[0])

sys.exit(0)

#要放到神經網路計算所以先進行規畫
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]#引入房價數據
scaled_data = scaler.fit_transform(housing.data)

data = np.c_[np.ones((m,1)), scaled_data]

n_epoch = 1000
learning_rate = 0.01

#設placeholder
X = tf.constant(data, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

#theta 可以理解為權重
#tf.random_uniform: 在圖中創建一個包含有隨機值的結點
#可理解為numpy中的random函數, 相當於初始的權重是隨機賦值的
theta = tf.Variable(tf.random_uniform([n+1,1],-1,1),name='theta')
y_pred = tf.matmul(X, theta, name='prediction')#X與權重相乘
error = y_pred-y#預測值和原始的label間的差值
mse = tf.reduce_mean(tf.square(error), name='mse')#Mean Squared Error

#計算梯度公式
gradient = 2/m * tf.matmul(tf.transpose(X), error)
#tf.assign: 創建一個將新值賦給變量的一個節點, 為Variable更新值
#相當於實現了之前權重更新結點公式的迭代
training_op = tf.assign(theta, theta-learning_rate*gradient)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        if epoch % 100 ==0:
            print("Epoch", epoch, "MSE = ", mse.eval())
    print("best theta", theta.eval())
