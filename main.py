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
    classifiedMasks=[]
    for image in maskImages:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        for pixelRow in image:
            for pixel in pixelRow:
                maskClass=classifyPixel(pixel)
                classifiedPixel=np.append(pixel,maskClass)
                classifiedMasks.append(classifiedPixel)
    return classifiedMasks
                
def classifyPixel(pixel):
    pixelList=pixel.tolist()
    match pixelList:
        case [61, 61, 245]:
            return 0
        case [221, 255, 51]:
            return 1
        case [255, 53, 94]:
            return 2
        case [255, 204, 51]:
            return 3
        case [184, 61, 245]:
            return 4
        case _:
            return 5
#=======================================================================================================================

imgs = loadDataset("train/images",True)

masks = loadDataset("train/masks")
masks=getMaskClass(masks)
