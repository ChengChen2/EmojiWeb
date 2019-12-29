# -*- coding: utf-8 -*-  
from PIL import Image  
import pytesseract  
import cv2
import numpy as np
from matplotlib import pyplot as plt 

img_num = 29

# OCR options 
custom_oem_psm_config = r'--psm 6 --oem 2'
custom_lang = 'chi_sim'

# ------------------------- Functions ---------------------------------------------
def kmeans(input_img, k, i_val):
    hist = cv2.calcHist([input_img],[0],None,[256],[0,256])
    img = input_img.ravel()
    img = np.reshape(img, (-1, 1))
    img = img.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(img,k,None,criteria,10,flags)
    centers = np.sort(centers, axis=0)

    return centers[i_val].astype(int), centers, hist
####################################################################################


for i in range(img_num):
    img = cv2.imread(str(i) + '.jpg')
    _, thresh = cv2.threshold(img, kmeans(input_img=img, k=8, i_val=2)[0], 255, cv2.THRESH_BINARY)
    print(str(i) + '. :\n')
    text = pytesseract.image_to_string(img, lang=custom_lang, config=custom_oem_psm_config) 
    print(text + '\n' + '-----------------------------\n') 