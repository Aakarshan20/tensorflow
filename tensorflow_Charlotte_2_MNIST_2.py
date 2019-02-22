#手寫數字識別
import tensorflow as tf

#加載mnist 數據集
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print(mnist)

#設置參數
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph輸入
x = tf.placeholder(tf.float32, [None, 784])#mnist的圖片大小
y = tf.placeholder(tf.float32, [None, 10])

#設定權重和偏移

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#設定運行模型
pred = tf.nn.softmax(tf.matmul(x, W)+b) #使用softmax來預測

#設置cost function為交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化權重
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历每个batch
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 把每个batch的数据放进去训练
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # 计算平均损失
            avg_cost += c / total_batch
        # 展示每次迭代的日志
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算3000个样本的准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))








