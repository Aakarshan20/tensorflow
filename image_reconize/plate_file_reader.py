import tensorflow as tf 

# 新建一个Session
with tf.Session() as sess:
    #要讀的圖片plate1.jpg, plate2.jpg, plate3.jpg
    filename = ['images/plate1.jpg', 'images/plate2.jpg', 'images/plate3.jpg']
    # string_input_producer 產生文件名隊列
    #num_epoch 為迭代次數 此處為五 代表共要讀15次
    #shuffle=True代表會先打亂filename 然後再進filename的隊列中
    filename_queue = tf.train.string_input_producer(filename, shuffle=True, num_epochs=5)
    
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tf.train.string_input_producer中有 epoch 所以要初始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之後才會填充隊列 不然會阻塞
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        # 讀圖加存圖
        image_data = sess.run(value)
        with open('tfIO/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
            print(filename_queue)
        #if i==3:
            #break

