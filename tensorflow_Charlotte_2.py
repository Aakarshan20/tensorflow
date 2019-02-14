import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

#參數設定
learning_rate = 0.01
training_epochs = 10000#迭代次數
display_stop = 50#每隔50次迭代 會打印出數字

#訓練模型(用來預測的樣本)
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.197,0.42,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.336,2.596,2.53,1.2,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples = train_X.shape[0]

print("train_X:", train_X)
print("train_Y:", train_Y)

#設置placeholder
X = tf.placeholder("float")
Y = tf.placeholder("float")

#設置模型的權重和偏置
#W和B在模型訓練中是會不斷更新的 所以用 Variable來保存
#最開始的權重和偏置是隨機的
W = tf.Variable(rng.randn(), name="weight") 
b = tf.Variable(rng.randn(), name="bias")

#設置線性迴歸的方程 也就是增加兩個結點
#y = Wx +b
#pred代表預測的值
pred = tf.add(tf.multiply(X, W), b)

#設置線性迴歸的均方差
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

#梯度下降
#注意 minimize() 可以修正W 和 b 因為默認設置Variables的trainable=True
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化所有variables
init = tf.global_variables_initializer()

#開始訓練
with tf.Session() as sess:
    sess.run(init)

    #灌入所有訓練數據(樣本)
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X, train_Y):#zip 打包可迭代的list 一一對應且元素個數以最短的為準
            sess.run(optimizer, feed_dict={X:x, Y:y})

            #打印出每次迭代的log
            if(epoch+1) % display_step == 0:





