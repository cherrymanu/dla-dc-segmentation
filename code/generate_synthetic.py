"""
Generate synthetic multi-block 'document' images for experiments.
"""

import cv2
import numpy as np
import random

def generate_synthetic_document(width=800, height=1000, seed=None):
    if seed: random.seed(seed)
    img = 255*np.ones((height,width),dtype=np.uint8)
    # Random rectangles: text, table, image, blank
    for _ in range(random.randint(5,10)):
        w,h = random.randint(100,300), random.randint(60,150)
        x = random.randint(0,width-w-1)
        y = random.randint(0,height-h-1)
        type_ = random.choice(["text","table","image"])
        if type_=="text":
            for i in range(y,y+h,random.randint(8,12)):
                cv2.line(img,(x,i),(x+w,i),0,1)
        elif type_=="table":
            for i in range(y,y+h,random.randint(20,40)):
                cv2.line(img,(x,i),(x+w,i),0,1)
            for j in range(x,x+w,random.randint(20,40)):
                cv2.line(img,(j,y),(j,y+h),0,1)
        else:
            noise = np.random.randint(0,100,(h,w),dtype=np.uint8)
            img[y:y+h,x:x+w]=cv2.addWeighted(img[y:y+h,x:x+w],0.5,noise,0.5,0)
    return img

def save_document(img, path):
    cv2.imwrite(path,img)
