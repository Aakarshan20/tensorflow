import tensorflow as tf

B = tf.Variable(0, dtype=tf.int64)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(5):
        print(sess.run(B.assign(i)))
        #下例無法正常賦值(未sess.run)
        #B.assign(i)
        #print(sess.run(B))
        
