from enum import Enum
import os
import numpy as np
import matplotlib.pyplot as plt
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
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)        
        classifiedImage = np.concatenate([image, classes], axis=2)
        classifiedMasks.append(classifiedImage)
    
    return classifiedMasks

def naiveBayes(X,targetClass,targetName,y,method,bins):
    classMask=(y==targetClass)
    classHues=X[classMask]
    nonClassMask=(y!=targetClass)
    nonClassHues=X[nonClassMask]
    model={}
    nonTargetName="non"+targetName
    if method==bayesType.Gaussian:
            model={
                targetName:{
                    "prior":len(classHues)/len(X),
                    "mean":np.mean(classHues),
                    "var":np.var(classHues)
                },
                nonTargetName:{
                    "prior":len(nonClassHues)/len(X),
                    "mean":np.mean(nonClassHues),
                    "var":np.var(nonClassHues)
                }
            }
            def likelihood(x, mean, var):
                return (1 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean)**2 / (2 * var))
    else:
        classHist, classEdges = np.histogram(classHues, bins=bins, range=(0, 360), density=True)
        nonClassHist, nonClassEdges = np.histogram(nonClassHues, bins=bins, range=(0, 360), density=True)
        centers = 0.5 * (classEdges[:-1] + classEdges[1:])
        
        model = {
            "road": {
                "prior": len(classHues) / len(X),
                "hist": classHist,
                "centers": centers
            },
            "non_road": {
                "prior": len(nonClassHist) / len(X),
                "hist": nonClassHist,
                "centers": centers
            }
        }
        def likelihood(x, hist, centers):
            idx = np.searchsorted(centers, x, side="right") - 1
            # idx = np.clip(idx, 0, len(hist) - 1)
            return hist[idx]
    return model, likelihood

def segmentImage(imagePath,targetName, model, likelihood, method):
    nonTargetName="non"+targetName
    imageBGR = cv2.imread(imagePath)
    imgHSV = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2HSV)
    
    hue = imgHSV[:, :, 0].astype(np.float32) * 2
    H, W = hue.shape
    
    testX = hue.flatten()
    
    if method == bayesType.Gaussian:
        targetP = model[targetName]["prior"] * likelihood(testX, model[targetName]["mean"], model[targetName]["var"])
        nonTargetP = model[nonTargetName]["prior"] * likelihood(testX, model[nonTargetName]["mean"], model[nonTargetName]["var"])
    else:
        targetP = model[targetName]["prior"] * likelihood(testX, model[targetName]["hist"], model[targetName]["centers"])
        nonTargetP = model[nonTargetName]["prior"] * likelihood(testX, model[nonTargetName]["hist"], model[nonTargetName]["centers"])
    
    targetMask = (targetP > nonTargetP).astype(np.uint8)
    targetMask = targetMask.reshape(H, W)
    
    result = targetMask * 255
    
    return result
#=======================================================================================================================

imgs = loadDataset("train/images",True)
imgsNP=np.array(imgs)

hues=imgsNP[:, :, :, 0].astype(np.float32) * 2

masks = loadDataset("train/masks")
masks=getMaskClass(masks)
masksNP=np.array(masks)

X = hues.flatten()  
y = masksNP.flatten()

method = bayesType.Histogram  
bins = 50
targetName="road"    
model, likelihood = naiveBayes(X,0,targetName, y, method, bins)

testImg = "test.jpg"

if os.path.exists(testImg):
    
    result = segmentImage(testImg, targetName, model, likelihood, method)
        
    plt.figure(figsize=(15, 5))
         
    plt.plot()
    plt.imshow(result, cmap='gray')
    plt.title("Road Segmentation\n(White = Road, Black = Non-road)")
        
    plt.show()