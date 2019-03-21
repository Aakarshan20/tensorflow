import tensorflow as tf

a = tf.add(2,3)
b = tf.multiply(a,4)

with tf.Session() as sess:
    replace_dict = {a:15}
    print(sess.run(b))#, feed_dict=replace_dict))
