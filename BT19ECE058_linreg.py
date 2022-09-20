#!/usr/bin/env python
# coding: utf-8

# In[687]:


import pandas as pd
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

path = 'E:\\ML COURSE VNIT\\A2\\Matlab_accidents.mat'
mat = scipy.io.loadmat(path)
data = mat['accidents']


# In[689]:


states = data[0][0][1] # State names
num_data = data[0][0][2] # Numerical data
columns = data[0][0][3][0] # Column Names


# In[690]:


cols = [] # Column Names
for i in range(0,len(columns)):
    cols.append(columns[i][0])


# In[691]:


data_val = data[0][0][2]
df_dict = {}
for i in range(0,len(cols)):
    val = []
    for j in range(0,len(num_data)):
        val.append(num_data[j][i])
    df_dict[cols[i]] = val


# In[692]:


df = pd.DataFrame(df_dict)
df.insert(0, 'State', states)


# In[693]:


df


# In[694]:


path_store = 'E:\\ML COURSE VNIT\\A2\\us_acc.csv'
df.to_csv(path_store,index=False)


# In[695]:


ratio = 0.75


# In[696]:


def BT9ECE065_dataset_div_shuffle(path,ratio): # The function created to split the data
    
    '''
    Here, we will check the extension of the dataset. If it's .mat file, 
    then, we process it in different way
    '''
    str_len = len(path)
    ext_of_path = path[str_len-4:str_len]
    
    if ext_of_path == '.mat':
        
        # Loading the .mat file using scipy library (returns in a dictionary format)
        mat = scipy.io.loadmat(path)
        
        
        list_of_data = []
        for i in mat.keys():
            ''' Here, we will store data whose keys are not identifiers 
                i.e, key name doesn't start with an "underscore" (like __val__)'''
            if i[0]!="_":
                list_of_data.append(mat[i])
                
        ''' Now, we convert first component of list to dataframe and 
            next, we will append other elements of dictionary to this dataframe.
            Since, in .mat file, the data and labels are stored in dictionary format
            (with seperate keys), we take all of them in a combined dataframe.
            '''
        df = pd.DataFrame(list_of_data[0])
        for i in range(1,len(list_of_data)):
            val = pd.DataFrame(list_of_data[i])
            for j in range(0,len(list_of_data[i][0])):
                str_val = 'val_'+str(i)+'_'+str(j)
#                 print("lol:",val[j])
                df[str_val] = pd.DataFrame(val[j])
                
        # df is our required dataframe
    
    else:
        # Reading the dataset if it's ".csv or .xls" format 
        df = pd.read_csv(path)
    
    '''
    Using sample() function in pandas to split the dataset into train and test sets.
    "frac = ratio" splits that much ratio of data into train_set.
    The sample() function also shuffles the dataset.
    '''
    train_set = df.sample(frac=ratio,random_state=1,replace = True)
    
    
    '''
    To get testset, Drop the rows that are in train_set from entire dataframe
    and store the leftover rows. This is our test_set
    '''
    test_set = df.drop(train_set.index)
    
    return train_set,test_set


# In[697]:


TrainSet,TestSet = BT9ECE065_dataset_div_shuffle(path_store,ratio)
TrainSet = shuffle(TrainSet)


# In[698]:


train_x = TrainSet['Registered vehicles (thousands)']
train_y = TrainSet['Fatalities per 100K registered vehicles']


# In[699]:


# Y = theta*W 
theta = np.ones((len(train_x),2),dtype=np.long)
theta[:,0] = train_x

Y = np.zeros((len(train_y),1),dtype = np.uint8)
Y[:,0] = train_y
W = np.zeros((2,1)) 


# In[700]:


W = np.matmul(np.matmul(np.linalg.inv(np.matmul(theta.T, theta)),theta.T),Y)


# In[701]:


m = W[0]
c = W[1]


# In[702]:


Y_hat = m*theta[:,0] + c


# In[703]:


Y_hat


# In[704]:


plt.scatter(theta[:,0],Y)
plt.plot(theta[:,0],Y_hat,"--")
plt.show()


# In[705]:


# Testing
test_x = TestSet['Traffic fatalities']
test_y = TestSet['Fatalities involving high blood alcohol']


# In[706]:


# Y = theta*W 
theta_test = np.ones((len(test_x),2),dtype=np.long)
theta_test[:,0] = test_x

Y_test = np.zeros((len(test_y),1),dtype = np.uint8)
Y_test[:,0] = test_y


# In[707]:


Y_hat_test = m*theta_test[:,0] + c


# In[708]:


error = np.square(np.subtract(Y_test[0],Y_hat_test[0])).mean()


# In[709]:


error


# In[ ]:





# In[ ]:





# In[723]:


# USING GRADIENT DESCENT
L = 1e-10
epochs = 5000
n = len(theta[:,0])
n


# In[724]:


import random


# In[725]:


m,c = random.random(),random.random()


# In[726]:


m,c


# In[727]:


for i in range(epochs): 
    Y_hat = m*theta[:,0] + c  # The current predicted value of Y
    l = np.matmul(theta[:,0],(Y - Y_hat))
    D_m = (-2/n) * np.sum(l)  # Derivative wrt m
    D_c = (-2/n) * np.sum(Y - Y_hat)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)


# In[733]:


Y_hat_test_g = m*theta_test[:,0] + c


# In[734]:


error = np.square(np.subtract(Y_test[0],Y_hat_test_g[0])).mean()


# In[735]:


error


# In[ ]:





# In[ ]:





# In[738]:


# Using another relationship between input and output

'''
parameters are Total Population, Licensed drivers, Registered Vehicles and Let's predict Traffic fatalities
'''


# In[739]:


TrainSet


# In[740]:


train_x1 = TrainSet['Total Population']
train_x2 = TrainSet['Licensed drivers (thousands)']
train_x3 = TrainSet['Registered vehicles (thousands)']
train_y1 = TrainSet['Traffic fatalities']


# In[747]:


# Y = theta*W 
theta_val = np.ones((len(train_x1),4),dtype=np.long)
theta_val[:,0] = train_x1
theta_val[:,1] = train_x2
theta_val[:,2] = train_x3

Y_val = np.zeros((len(train_y1),1),dtype = np.uint8)
Y_val[:,0] = train_y1
W_opt= np.zeros((4,1)) 


# In[749]:


W_opt = np.matmul(np.matmul(np.linalg.inv(np.matmul(theta_val.T, theta_val)),theta_val.T),Y_val)


# In[751]:


Y_hat_val = W_opt[0]*train_x1 + W_opt[1]*train_x2 + W_opt[2]*train_x3 + W_opt[3]


# In[756]:


plt.scatter(theta_val[:,0],Y_val)
plt.plot(theta_val[:,0],Y_hat_val,"--")
plt.show()


# In[757]:


plt.scatter(theta_val[:,1],Y_val)
plt.plot(theta_val[:,1],Y_hat_val,"--")
plt.show()


# In[762]:


test_x1 = TestSet['Total Population']
test_x2 = TestSet['Licensed drivers (thousands)']
test_x3 = TestSet['Registered vehicles (thousands)']
test_y1 = TestSet['Traffic fatalities']


# In[768]:


# Y = theta*W 
theta_val_test = np.ones((len(test_x1),4),dtype=np.long)
theta_val_test[:,0] = test_x1
theta_val_test[:,1] = test_x2
theta_val_test[:,2] = test_x3

Y_val_test = np.zeros((len(test_y1),1),dtype = np.uint8)
Y_val_test[:,0] = test_y1


# In[769]:


Y_hat_val_test = W_opt[0]*test_x1 + W_opt[1]*test_x2 + W_opt[2]*test_x3 + W_opt[3]


# In[770]:


error = np.square(np.subtract(Y_val_test[0],Y_hat_val_test)).mean()


# In[771]:


error


# In[ ]:




