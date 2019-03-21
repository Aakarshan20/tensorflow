import tensorflow as tf

my_var = tf.Variable(0, trainable=False)

init = tf.global_variables_initializer()

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(init)
print(sess1.run(my_var.assign_add(1)))

sess2.run(init)
print(sess2.run(my_var.assign_add(2)))



print(sess1.run(my_var.assign_add(5)))
print(sess2.run(my_var.assign_add(2)))
