import cv2
import os
import numpy as np
#I1 = cv2.imread('./images/chihuaha.png')

#I2 = cv2.imread('./images/chihuahua.png', 0)
'''
for dirPath, dirNames, fileNames in os.walk(".\images"):
    print(dirPath)
    for f in fileNames:
        print(os.path.join(dirPath, f))
'''


def get_File(file_dir):
    images = []

    subfolders = []

    for dirPath, dirNames, fileNames in os.walk(file_dir):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))
        for name in dirNames:
            subfolders.append(os.path.join(dirPatn, name))
    return images, subfolders
    
    labels = []

    count = 0
    
    for a_folder in subfolders:
        n_img = len(os.listdir(a_folder))
        labels = np.append(labels, n_img * [count])
        count+=1
    
    subfolders = np.array([images, labels])
    usbfolders = subfolders.transpose()
    
    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])

    label_list = [int(float(i)) for i in label_list]
    
    return image_list, label_list
    

print(get_File('.\images'))
