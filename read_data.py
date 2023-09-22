#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


def read_data():
    df=pd.read_csv('https://drive.google.com/u/0/uc?id=1R1w1K9gzfMKyDjG9SpoTu0KsC48Q1XP5&export=download')   #for preprocessing
    df1=pd.read_csv('https://drive.google.com/u/0/uc?id=1R1w1K9gzfMKyDjG9SpoTu0KsC48Q1XP5&export=download')  #for returning results
    return df.iloc[:100,:]

