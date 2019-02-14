import tensorflow as tf

C = tf.placeholder(dtype=tf.int64)
D = tf.placeholder(dtype=tf.int64)
E = tf.placeholder(dtype=tf.int64)

F = D+E

with tf.Session() as sess:
    result = sess.run(F, feed_dict={C:10, D:20, E:30})
    print(result)

    result = sess.run(F, feed_dict={D:20, E:30})
    print(result)

    result = sess.run(F, feed_dict={E:30})
    print(result)

