import tensorflow as tf
import numpy as np


g = tf.Graph()


with g.as_default():
    c = tf.constant(30.0)
assert c.graph is g
indefaultgraph = tf.add(1, 2, name="indefaultgraph")
        
with g.as_default():
    ingraphg = tf.multiply(2,3,name="ingraphg")
    alsoindefgraph = tf.subtract(5, 1, name='alsoindef')

writer = tf.summary.FileWriter('./my_graph', g)
    
     
