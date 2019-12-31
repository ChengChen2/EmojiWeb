# -*- coding: utf-8 -*-  
from PIL import Image  
import pytesseract  
import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt 

# Static values
pixel_thresh_ub = 235
pixel_thresh_lb = 20
img_path = "all_class"

# OCR options 
custom_oem_psm_config_0 = r'--psm 6 --oem 2'
custom_oem_psm_config_1 = r'--psm 11 --oem 2'
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
    
def img_text_detection(input_img, img_name):
    img_type = img_histogram_classify(input_img)
    if(img_type == 0):
        text = pytesseract.image_to_string(input_img, lang=custom_lang, config=custom_oem_psm_config_0)
    elif(img_type == 1):
        _, thresh = cv2.threshold(input_img, kmeans(input_img, k=8, i_val=2)[0], 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, lang=custom_lang, config=custom_oem_psm_config_1)
    else:
        text = 'classificaiton error - unknow image type'
    
    print(img_name + '. :\n')
    print(text + '\n##########################\n') 
    plt.imshow(input_img)
    plt.title('Img:' + img_name + ' Type:' + str(img_type))
    plt.show()
    
def img_histogram_classify(input_img):
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    pixel_values = img_gray.ravel()
    pixel_max = np.argmax(np.bincount(pixel_values))
    if(pixel_max >= pixel_thresh_ub or pixel_max <= pixel_thresh_lb):
        return 0
    else:
        return 1 
        

    
####################################################################################


# ---------------------------- Main ---------------------------------
for filename in os.listdir(img_path):
    if filename.endswith('jpg') or filename.endswith('png'):
        img = cv2.imread(filename)
        if(img is None):
            img = cv2.imdecode(np.fromfile(img_path + "/" + filename, dtype=np.uint8), -1)
            if(img is None):
                print("wrong img format")
            else:
                img_text_detection(img, filename)
        else:
            img_text_detection(img, filename)
            
    
    