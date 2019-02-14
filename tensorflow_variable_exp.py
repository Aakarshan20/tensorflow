import tensorflow as tf

state = tf.Variable(0, name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state ,one)

update = tf.assign(state, new_value)

#因為用tf.Variable來定義變量, 所以要用以下方法來初始它們(但尚未啟用)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
