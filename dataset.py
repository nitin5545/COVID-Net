#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# In[3]:


class COVID19(Dataset):
    def __init__(self, df, tf):
        self.df = df
        self.tf = tf
        
    def __len__(self):
        return len(df)
    
    def __getitem__(self, idx):
        _, img_path, labels = self.df.iloc[idx]
        
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert("RGB")
            
            if self.tf:
                image = self.tf(image)
            
            return image, labels
        
        except:
            print(img_path)
        


# In[ ]:




