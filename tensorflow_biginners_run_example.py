import tensorflow as tf

a = tf.constant([10,20])
b = tf.constant([1.0, 2.0])

sess = tf.Session()

#v = sess.run(a)
v = sess.run([a,b])

print(v)
