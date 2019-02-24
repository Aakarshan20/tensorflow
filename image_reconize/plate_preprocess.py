import tensorflow as tf
import matplotlib.pyplot as plt


image_raw_data = tf.gfile.FastGFile('images/plate1.jpg', 'rb').read()


def show(img_data):
    plt.imshow(img_data.eval())
    plt.show()

with tf.Session() as sess:
    #將原始數據解碼成多維矩陣
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())
    show(img_data)

    #將圖像的矩陣編碼成圖像並存入文件
    encoded_image = tf.image.encode_jpeg(img_data)
    #在存之前要建好資料夾 避免error
    with tf.gfile.GFile('preprocess/output.jpg', 'wb') as f:
        f.write(encoded_image.eval())

    #將圖像數據的類型轉為實數類型 便於對圖像進行處理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
