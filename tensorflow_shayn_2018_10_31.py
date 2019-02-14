import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

A = tf.constant("Hello World!")
A = tf.constant("Bye World!")#, dtype=tf.int64)

with tf.Session() as sess:

    print(A)
    print(sess.run(A))
    
    #B=sess.run(A)
    #print(B)
