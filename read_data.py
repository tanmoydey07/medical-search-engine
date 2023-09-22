#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


def read_data():
    df=pd.read_csv('Dimension-covid.csv')   #for preprocessing
    df1=pd.read_csv('Dimension-covid.csv')  #for returning results
    return df.iloc[:100,:]

