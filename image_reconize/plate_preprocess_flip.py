import tensorflow as tf
import matplotlib.pyplot as plt


image_raw_data = tf.gfile.FastGFile('images/plate1.jpg', 'rb').read()


def show(img_data):
    plt.imshow(img_data.eval())
    plt.show()

#resize_images 調整圖像大小
#第一個參數為原始圖像
#第二個參數為調整後的圖像大小[new_height, new_weight]跟舊版分為兩個參數不同
#method代表調整圖像大小的算法
#0: 雙線性差值法, 1:最近鄰居法, 2:雙三次差值法, 3:面積差值法

with tf.Session() as sess:
    #將原始數據解碼成多維矩陣
    img_data = tf.image.decode_jpeg(image_raw_data)
    print('原始圖像')
    show(img_data)

    #圖像翻轉
    flipped = tf.image.flip_up_down(img_data)#上下
    print('上下翻轉')
    show(flipped)

    flipped = tf.image.flip_left_right(img_data)#左右
    print('左右翻轉')
    show(flipped)

    transposed = tf.image.transpose_image(img_data)#對角線
    print('對角線翻轉')#也就是沿左上到右下那條軸 將左下的點放到右上
    show(transposed)

    transposed = tf.image.transpose_image(transposed)#對角線
    print('對角線翻轉')#也就是沿左上到右下那條軸 將左下的點放到右上
    show(transposed)

    flipped = tf.image.random_flip_up_down(img_data)#隨機上下
    print('隨機上下翻轉')
    show(flipped)

    flipped = tf.image.random_flip_left_right(img_data)#隨機左右
    print('隨機左右翻轉')
    show(flipped)





    
