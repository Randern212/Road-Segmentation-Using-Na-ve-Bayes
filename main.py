from enum import Enum
import os
import numpy as np
import pandas as pd
import cv2
class  bayesType(Enum):
    Gaussian=0
    Histogram=1

class segment(Enum):
    roadSurface=0
    marking=1
    roadSign=2
    car=3
    background=4

#=======================================================================================================================
#images are loaded as img[row][column]
def loadDataset(path):
    imgs = []
    for root,dirs,files in os.walk(path):
        for imgName in files:
            imgPath = os.path.join(root,imgName)
            if imgName.lower().endswith(('.png', '.jpg', '.jpeg')):
                imgBGR = cv2.imread(imgPath)
                imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
                imgs.append(imgHSV)
    return imgs
#=======================================================================================================================

masks=loadDataset("train/masks")
imgs=loadDataset("train/images")

