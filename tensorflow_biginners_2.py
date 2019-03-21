import tensorflow as tf
import numpy as np


nt = [4,3]
npt = np.array([4,3], dtype=np.int32)
a = tf.constant(npt, name="constant_a")
b = tf.reduce_sum(a, name="sum_b")
c = tf.reduce_prod(a, name="mul_c")
d = tf.multiply(b,c, name="mul_d")

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./my_graph', sess.graph)
