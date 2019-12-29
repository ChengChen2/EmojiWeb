# -*- coding: utf-8 -*-  
from PIL import Image  
import pytesseract
import numpy as np
from matplotlib import pyplot as plt   
import cv2
import os 

#np.set_printoptions(threshold=np.inf)

img_num = 0

img_path = "class1"
for filename in os.listdir(img_path):
    if filename.endswith('jpg') or filename.endswith('png'):
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        values = img_gray.ravel()
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.hist(values, bins=256, range=[0,256])
        plt.title('figure: '+ str(img_num))
        img_num += 1
        plt.show()
    
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