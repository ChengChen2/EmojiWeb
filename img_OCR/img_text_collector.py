# -*- coding: utf-8 -*-  
from PIL import Image  
import pytesseract  
import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt 

# OCR options 
custom_oem_psm_config = r'--psm 11 --oem 2'
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
    
def text_detection(img):
    _, thresh = cv2.threshold(img, kmeans(input_img=img, k=8, i_val=2)[0], 255, cv2.THRESH_BINARY)
    print(str(filename) + '. :\n')
    text = pytesseract.image_to_string(thresh, lang=custom_lang, config=custom_oem_psm_config) 
    print(text + '\n-------------kmeans----------------\n') 
    text = pytesseract.image_to_string(img, lang=custom_lang, config=custom_oem_psm_config) 
    print(text + '\n###############non-kmeans##############\n')
    
####################################################################################


img_path = "class1"
for filename in os.listdir(img_path):
    if filename.endswith('jpg') or filename.endswith('png'):
        img = cv2.imread(filename)
        if(img is None):
            img = cv2.imdecode(np.fromfile(img_path + "/" + filename, dtype=np.uint8), -1)
            if(img is None):
                print("wrong img format")
            else:
                text_detection(img)
        else:
            text_detection(img)
            
    
    