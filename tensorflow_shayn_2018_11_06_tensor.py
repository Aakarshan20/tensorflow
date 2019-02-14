import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

A = tf.constant(50)
B = tf.constant(100)

C = A+B

print(C)
