import tensorflow as tf

a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 6.0])

weights = tf.Variable(tf.random_normal([784,200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")



#v1 = tf.variables_initializer([a,b], name="v1")

x1 = tf.placeholder(tf.int32, shape=[1], name='x1')
x2 = tf.constant(2, name='x2')

result = x1+x2

with tf.Session() as sess:
    #print(sess.run(result))
    print(sess.run(result, {x1:[3]}))
        
