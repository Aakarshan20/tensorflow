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
#print(housing.keys())
#print(housing.feature_names)
#print(housing.target)
#print(housing.DESCR)
print(housing.data[0])

sys.exit(0)

housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
scaled_data = scaler.fit_transform(housing.data)
data = np.c_[np.ones((m,1)), scaled_data]

n_epoch = 1000
learning_rate = 0.01

#設placeholder
X = tf.constant(data, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n+1,1],-1,1),name='theta')
y_pred = tf.matmul(X, theta, name='prediction')
error = y_pred-y
mse = tf.reduce_mean(tf.square(error), name='mse')#Mean Squared Error

#計算梯度公式
gradient = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta-learning_rate*gradient)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        if epoch % 100 ==0:
            print("Epoch", epoch, "MSE = ", mse.eval())
    print("best theta", theta.eval())
