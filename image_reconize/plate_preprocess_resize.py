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

    #將圖像的矩陣編碼成圖像並存入文件
    encoded_image = tf.image.encode_jpeg(img_data)
    #在存之前要建好資料夾 避免error
    with tf.gfile.GFile('preprocess/output.jpg', 'wb') as f:
        f.write(encoded_image.eval())

    #將圖像數據的類型轉為實數類型 便於對圖像進行處理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    resized = tf.image.resize_images(img_data, [300,300], method=0)
    print(resized.get_shape())#圖像深度如沒有顯示指定則為問號
    print('裁剪後的圖像')
    show(resized)

    #用resize_image_with_crop_or_pad調整大小
    #第一個參數為原始圖像
    #第二個和第三個參數是調整後的圖像大小, 大於原圖則填充, 小於則裁剪居中部份
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 200, 200)
    print('通過resize_image_with_crop_or_pad處理後的圖像, 此圖為裁切後')
    show(croped)

    padded = tf.image.resize_image_with_crop_or_pad(img_data, 500, 700)
    print('通過resize_image_with_crop_or_pad處理後的圖像, 此圖為填充後')
    show(padded)

    #用central_crop調整圖像大小
    #第一個參數是原始圖像
    #第二個參數為調整比例 是(0,1]的實數
    central_cropped = tf.image.central_crop(img_data, 0.5)
    print('通過比例裁剪後的圖像')
    show(central_cropped)







    
