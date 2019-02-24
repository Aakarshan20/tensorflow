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
    #show(img_data)
    
    #調整圖像的亮度
    adjusted = tf.image.adjust_brightness(img_data, 0.5)#亮度+0.5
    print('調整圖像的亮度')
    show(adjusted)

    #範圍內[-0.5, 0.5]隨機調整圖像亮度
    adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
    print('[-0.5-0.5]區間內隨機調整圖像亮度')
    show(adjusted)

    #調整對比度
    adjusted = tf.image.adjust_contrast(img_data, -5)#將圖像的對比度-5
    print('對比度-5')
    show(adjusted)
    adjusted = tf.image.adjust_contrast(img_data, 15)#將圖像的對比度+5
    print('對比度+5')
    show(adjusted)
    #lower 不得為負 會error
    adjusted = tf.image.random_contrast(img_data, lower=1.5, upper=5)#隨機調整對比度
    print('隨機對比度')
    show(adjusted)

    #調整飽和度
    adjusted = tf.image.adjust_saturation(img_data, -5)#將圖像的飽和度-5
    print('飽和度-5')
    show(adjusted)
    adjusted = tf.image.adjust_saturation(img_data, 15)#將圖像的飽和度+5
    print('飽和度+5')
    show(adjusted)
    #lower 不得為負 會error
    adjusted = tf.image.random_saturation(img_data, lower=-1, upper=5)#隨機調整飽和度
    print('隨機飽和度')
    show(adjusted)


    #調整色相
    adjusted = tf.image.adjust_hue(img_data, 0.5)#將色相+0.5
    print('將色相+0.5')
    show(adjusted)
    adjusted = tf.image.random_hue(img_data, max_delta=0.5)#隨機調整色相
    print('隨機調整色相')
    show(adjusted)
    


















    





    





    
