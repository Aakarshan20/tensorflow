import tensorflow as tf

my_var = tf.Variable(4, name="my_variable")
add = tf.add(5, my_var)
mul = tf.multiply(8, my_var)

zeros = tf.zeros([3,2])
ones = tf.ones([4])
uniform = tf.random_uniform([4,4], minval=0, maxval=10)
normal = tf.random_normal([4,3,2], mean=0.0, stddev=3.0)

random_var = tf.Variable(tf.truncated_normal([2,2]))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(mul))
    #print(sess.run(add))
    sess.run(my_var)
    print(my_var.eval(session=sess))
    print(sess.run(zeros))
    print(sess.run(ones))
    print(sess.run(uniform))
    print(sess.run(normal))
    print(sess.run(random_var.initializer))
