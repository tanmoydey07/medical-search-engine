
#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#defining function to define cosine similarity
from numpy import dot
from numpy.linalg import norm
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
import pandas as pd

import numpy as np
import gensim
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText  

from matplotlib import pyplot
import string # used for preprocessing
import re # used for preprocessing
import nltk # the Natural Language Toolkit, used for preprocessing
import numpy as np # used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


from read_data import read_data
from preprocessing import preprocessing_input
from return_embed import get_mean_vector
import pickle

def cos_sim(a,b):

    return dot(a, b)/(norm(a)*norm(b)) 
#function to return top n similar results


def top_n(query,model_name,column_name):
    vector_size=100
    window_size=3
    df=read_data()
    if model_name=='Skipgram':
        
   
        word2vec_model=Word2Vec.load('model_Skipgram.bin')
        K=pd.read_csv('Skipgram_vec.csv')
    else:
        
        word2vec_model=Word2Vec.load('model_Fasttext.bin')
        K=pd.read_csv('Fasttext_vec.csv')
    #input vectors
    query=preprocessing_input(query)
    
    query_vector=get_mean_vector(word2vec_model,query)
    #Model Vectors
      #Loading our pretrained vectors of each abstracts

    p=[]                          #transforming dataframe into required array like structure as we did in above step
    for i in range(df.shape[0]):
        p.append(K[str(i)].values)    
    x=[]
    #Converting cosine similarities of overall data set with input queries into LIST
    for i in range(len(p)):
        x.append(cos_sim(query_vector,p[i]))
    
    
 #store list in tmp to retrieve index
    tmp=list(x)
    
 #sort list so that largest elements are on the far right
    
    res = sorted(range(len(x)), key = lambda sub: x[sub])[-10:]
    sim=[tmp[i] for i in reversed(res)]
    
 #get index of the 10 or n largest element
    L=[]
    for i in reversed(res):
    
        L.append(i)
        
    df1=read_data()    
    return df1.iloc[L, [1,2,5,6]],sim     #returning dataframe (only id,title,abstract ,publication date)
