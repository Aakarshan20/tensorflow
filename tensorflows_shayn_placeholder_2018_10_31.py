import tensorflow as tf

C = tf.placeholder(dtype=tf.int64)

with tf.Session() as sess:
    for i in range(5):
        result = sess.run(C, feed_dict={C:i})
        print(result)
