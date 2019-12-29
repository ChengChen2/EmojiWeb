# -*- coding: utf-8 -*-  
from PIL import Image  
import pytesseract
import numpy as np
from matplotlib import pyplot as plt   
import cv2
import os 

img_num = 0
#np.set_printoptions(threshold=np.inf)

def img_histogram(input_img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    values = img_gray.ravel()
    print(np.var(values))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.hist(values, bins=256, range=[0,256])
    plt.title('Histogram figure: '+ str(img_num))
    plt.show()


img_path = "class2"
for filename in os.listdir(img_path):
    if filename.endswith('jpg') or filename.endswith('png'):
        img = cv2.imread(filename)
        if(img is None):
            img = cv2.imdecode(np.fromfile(img_path + "/" + filename, dtype=np.uint8), -1)
            if(img is None):
                print("wrong img format")
            else:
                img_histogram(img)
                img_num += 1
        else:
            img_histogram(img)
            img_num += 1
    
'''
for i in range(img_num):
    img = cv2.imread(str(i) + '.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    values = img_gray.ravel()
    print(values.shape)
    print(values)
    plt.hist(values, bins=256, range=[0,256])
    plt.title('figure: '+ str(i))
    plt.show()
    
'''