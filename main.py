from enum import Enum
import os
import numpy as np
import pandas as pd
import cv2
class  bayesType(Enum):
    Gaussian=0
    Histogram=1

#=======================================================================================================================
#images are loaded as img[row][column]
def loadDataset(path,hsv:bool=False):
    imgs = []
    for root,dirs,files in os.walk(path):
        for imgName in files:
            imgPath = os.path.join(root,imgName)
            if imgName.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(imgPath)
                if hsv:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                imgs.append(img)
    return imgs
        
def getMaskClass(maskImages):
    classifiedMasks = []
    color_map = {
        (61, 61, 245): 0,
        (221, 255, 51): 1, 
        (255, 53, 94): 2,
        (255, 204, 51): 3,
        (184, 61, 245): 4
    }
    
    for image in maskImages:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        classes = np.full((h, w, 1),5, dtype=np.uint8)
        
        for color, class_id in color_map.items():
            mask = np.all(image == color, axis=2)
            classes[mask] = class_id
                
        classifiedImage = np.concatenate([image, classes], axis=2)
        classifiedMasks.append(classifiedImage)
    
    return classifiedMasks
#=======================================================================================================================

imgs = loadDataset("train/images",True)

masks = loadDataset("train/masks")
masks=getMaskClass(masks)
