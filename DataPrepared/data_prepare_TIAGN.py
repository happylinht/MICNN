#!/usr/bin/env python
# coding: utf-8
"""
This program is used to prepare the training data sets and test set for MICNN. 
TIAGN algorithm is used to balance the pulsars and non-pulsars in the training set.
"""
# In[22]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale


# In[2]:

path = '/home/DM11/Four_plots_HTRU.pkl'


# In[3]:


data = pickle.load(open(path,'rb'),encoding='latin1')  # import the four 


# In[4]:


print(data.keys())
y_name = list(data.keys())[1]
data_name = list(data.keys())[0]
X = data[data_name]
y =data[y_name]


# In[115]:


X_DM = np.array(list(X.DM))

X_subband = np.array(list(X.subbands)).reshape(-1,16*64)  # 91192*16*64

X_subint = np.array(list(X.time_vs_phase)).reshape(-1,16*64) # 91192*16*64

X_profile = np.array(list(X.sumprof))




# In[27]:


def statistic_feature(data):
    n=data.shape[0]
    statistic_result=[]

    for i in range(n):
        s = pd.Series(data[i,:])
        temp=[s.mean(),s.std(),s.skew(),s.kurt()]
        statistic_result.append(temp)
    return statistic_result


# In[37]:


Feature_2 = np.hstack([np.array(statistic_feature(X_DM)),np.array(statistic_feature(X_profile))]) #Feature_2 are eight statistic features


# In[40]:


Feature_2 = scale(Feature_2,axis=0)


# In[116]:


from sklearn.model_selection import train_test_split
np.random.seed(123)
X_int_train_1, X_int_test_1, y_train, y_test = train_test_split(X_subint,y,test_size=0.25)
np.random.seed(123)
X_int_train_2, X_int_test_2, y_train, y_test = train_test_split(Feature_2,y,test_size=0.25)


# In[145]:


def pulsar_transform_int(signal):  #平移
    #k= np.random.randint(1,16,1)[0]
    k = np.argmin(np.sum(signal.reshape(16,-1),axis=0)) 
    if k>32:
        k = k-32
    else:
        k = 32+k
    half1 = signal[k:]
    half2 = signal[0:k]
    signal_new = np.hstack([half1,half2])
    return signal_new

def pulsar_noise(signal):
    signal_new = np.random.rand(len(signal))*0.5+signal 
    return signal_new

def TIANG_int(data_1,data_2,label,imbalance_rate):     
    id_pulsar=np.where(label==1)[0]
    id_non_pulsar=np.where(label==0)[0]
    imbalance_rate =  len(id_non_pulsar)/len(id_pulsar)/imbalance_rate
    pulsar_num = int(imbalance_rate*len(id_pulsar)) # the number of samples for pulsar signals
    np.random.seed(123)
    id_sample_pulsar=id_pulsar[np.random.randint(0,len(id_pulsar),pulsar_num)]
    
    pulsar_data = data_1[id_sample_pulsar,:]
    pulsar_data_TIANG=0.*pulsar_data
    for j in range(len(pulsar_data)):
        pulsar_data_TIANG[j,:]=pulsar_noise(pulsar_transform_int(pulsar_data[j,:]))
    
    data_1_TIANG = np.vstack([pulsar_data_TIANG,data_1[id_non_pulsar,:]])
    data_2_TIANG = np.vstack([data_2[id_sample_pulsar,:],data_2[id_non_pulsar,:]])
    y_TIANG = np.hstack([label[id_sample_pulsar],label[id_non_pulsar]])
    
    return data_1_TIANG,data_2_TIANG,y_TIANG


# In[146]:



X_int_train_1_5,X_int_train_2_5, y_train_5= TIANG_int(X_int_train_1,X_int_train_2, y_train,imbalance_rate=5)  # the imbalance rate is 5:1
X_int_train_1_20,X_int_train_2_20, y_train_20= TIANG_int(X_int_train_1,X_int_train_2, y_train,imbalance_rate=20)  # the imbalance rate is 20:1


# In[148]:

np.save("X_int_train_1_5.npy",X_int_train_1_5)
np.save("X_int_train_2_5.npy",X_int_train_2_5)
np.save("y_train_5.npy",y_train_5)

np.save("X_int_train_1_20.npy",X_int_train_1_20)
np.save("X_int_train_2_20.npy",X_int_train_2_20)
np.save("y_train_20.npy",y_train_20)

np.save("X_int_train_1_75.npy",X_int_train_1)
np.save("X_int_train_2_75.npy",X_int_train_2)
np.save("y_train_75.npy",y_train)

np.save("X_int_test_1.npy",X_int_test_1)
np.save("X_int_test_2.npy",X_int_test_2)
np.save("y_test.npy",y_test)


# In[ ]:




