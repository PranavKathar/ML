#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
def BT19ECE058_dataset_div_shuffle(data_csv,training_ratio,testing_ratio):
    df=pd.read_csv(data_csv)
    print(df)
    print("");
    shuffled_data=df.sample(frac=1)
    rows=round(training_ratio*(df.shape[0]))
    training_data=shuffled_data.iloc[1:rows]
    testing_data=shuffled_data.iloc[rows:df.shape[0]]
    
    print(training_data)
    print("");
    print(testing_data)
    return


BT19ECE058_dataset_div_shuffle("score.csv",0.5,0.5)


# In[ ]:





# In[ ]:




