import os
import tensorflow as tf
from PIL import Image


def create_record():
    writer = tf.python_io.TFRecordWriter("tfIO/tfrecord/test.tfrecord")
    for i in range(3):
        # 创建字典
        features={}
        # 写入张量，类型float，本身是三维张量，另一种方法是转变成字符类型存储，随后再转回原类型
        # 要存入下列三種之一 int64 float32 string
        # 張量是無法做為feature list的 所以要進行轉換
        # 其中一種方法是轉換成list類型 將張量拍扁成list 再用寫入list的方式寫入
        # 本例是轉換成string類型 將張量用toString轉換
        features['tensor'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensors[i].tostring()]))
        # 存储形状信息(806,806,3) tensor_list是int 所以直接用int64存進去即可
        features['tensor_shape'] = tf.train.Feature(int64_list = tf.train.Int64List(value=tensors[i].shape))
        # 将存有所有feature的字典送入tf.train.Features中
        tf_features = tf.train.Features(feature= features)
        # 再将其变成一个样本example
        tf_example = tf.train.Example(features = tf_features)
        # 序列化该样本
        tf_serialized = tf_example.SerializeToString()
        # 写入一个序列化的样本
        writer.write(tf_serialized)
        # 由于上面有循环3次，所以到此我们已经写了3个样本
        # 关闭文件    
    writer.close()
    
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'tensor': tf.FixedLenFeature([], tf.string),
                                           'tensor_shape' : tf.FixedLenFeature([], tf.int64),
                                       })

    tensor = tf.decode_raw(features['tensor'], tf.uint8)
    tensor = tf.reshape(tensor, [224, 224, 3])
    tensor = tf.cast(tensor, tf.float32) * (1. / 255) - 0.5
    tensor_shape = tf.cast(features['tensor_shape'], tf.int32)

    return tensor,tensor_shape

if __name__ == '__main__':
    img, label = read_and_decode("tfIO/tfrecord/test.tfrecord")

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=5, capacity=2000,#一次灌入五張圖片 容量為2000
                                                    min_after_dequeue=1000)
    #初始化所有的op
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        #启动队列
        threads = tf.train.start_queue_runners(sess=sess)#把數據灌入內存
        for i in range(3):
            val, l= sess.run([img_batch, label_batch])
            #l = to_categorical(l, 12)
            print(val.shape, l)
