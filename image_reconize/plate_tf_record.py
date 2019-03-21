import tensorflow as tf
# 为显示图片
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
#%pylab inline
import pylab
# 为数据操作
import pandas as pd
import numpy as np
import os
img=mpimg.imread('images/plate1.jpg') 
tensors = np.array([img,img,img])
# show image
print('\n张量')
#display(tensors, show = False)
plt.imshow(img)
