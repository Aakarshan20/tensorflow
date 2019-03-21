import tensorflow as tf

my_var = tf.Variable(1)

my_var_plus_two = my_var.assign(my_var +2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(my_var))
    print(sess.run(my_var_plus_two))

    assign_op = my_var.assign(4)
    sess.run(assign_op)
    
    print(my_var.eval(session=sess))

    print(sess.run(my_var.assign_add(1)))

    print(sess.run(my_var.assign_sub(1)))
    
