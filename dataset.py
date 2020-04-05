#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# In[3]:

import cv2
def clache(img_path):
    clipLimit=2.0 
    tileGridSize=(4, 4)  
    bgr = cv2.imread(img_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab_planes[0] = clahe.apply(lab_planes[0]) 
    lab = cv2.merge(lab_planes)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


class COVID19(Dataset):
    def __init__(self, df, tf, cl):
        self.df = df
        self.tf = tf
        self.cl = cl
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        _, img_path, labels = self.df.iloc[idx]
        
        try:
            if self.cl:
                image = Image.fromarray(clache(img_path))
            else:
                image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert("RGB")
            
            if self.tf:
                image = self.tf(image)
            
            return image, labels
        
        except:
            print(img_path)
        


# In[ ]:




