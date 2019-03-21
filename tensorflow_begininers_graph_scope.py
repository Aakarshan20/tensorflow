import tensorflow as tf

with tf.name_scope("Scope1"):
    a = tf.add(1,2, name="Scope1_add")
    b = tf.multiply(a,2, name="Scope1_mul")

    
with tf.name_scope("Scope2"):
    c = tf.add(4,5, name="Scope2_add")
    d = tf.multiply(c,6, name="Scope2_mul")

e = tf.add(b, d, name="output")

writer = tf.summary.FileWriter('./name_scope', graph=tf.get_default_graph())
writer.close()


