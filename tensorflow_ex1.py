import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3



##create tensorflow structure start##
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))#權重


biases = tf.Variable(tf.zeros([1]))#偏移量

y = Weights*x_data + biases#預測的yWeights

loss = tf.reduce_mean(tf.square(y-y_data))#平方差

optimizer = tf.train.GradientDescentOptimizer(0.5)#0.5: learning rate

train = optimizer.minimize(loss)

#因為用tf.Variable來定義權重與偏移量, 所以要用以下方法來初始它們(但尚未啟用)
init = tf.global_variables_initializer()




##create tensorflow structure end##

sess = tf.Session()
sess.run(init)#啟用初始變量

for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step, sess.run(Weights), sess.run(biases))

        
