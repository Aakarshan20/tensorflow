import tensorflow as tf
import numpy as np


a = tf.placeholder(tf.int32, shape=[2], name="input")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b, c, name="add_d")

with tf.Session() as sess:
    input_dict = {a:np.array([1,2], dtype=np.int32)}
    print(sess.run(d, feed_dict=input_dict))

    writer = tf.summary.FileWriter('./my_graph', sess.graph)

    
     
