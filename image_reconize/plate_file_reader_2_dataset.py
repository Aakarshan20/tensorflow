import tensorflow as tf
#import tensorflow.contrib.eager as tfe
import numpy as np

'''
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]
'''
def _parse_function(filename):
    image_string = tf.read_file(filename)
    #image_decoded = tf.image.decode_jpeg(image_string)
    #image_resized = tf.image.resize_images(image_decoded, [28, 28])
    #return image_resized, label
    #return image_decoded
    return image_string


filenames = ['images/plate1.jpg', 'images/plate2.jpg', 'images/plate3.jpg']
dataset = tf.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(_parse_function)


iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    threads = tf.train.start_queue_runners(sess=sess)
    i=0
    while True:
        i+=1
        print(i)
        #print(sess.run(next_element))
        image_data = sess.run(next_element)
        print(image_data)

        #encoded_image = tf.image.encode_jpeg(image_data)
        
        with open('tfIO/test_%d.jpg' % i, 'wb') as f:
            #f.write(encoded_image)
            f.write(image_data)
        
'''
        try:
            i+=1
            print(i)
            print(sess.run(next_element))
            
        except tf.errors.OutOfRangeError:
            print("end")
            #sess.close()
        
'''
        






    





    





    
