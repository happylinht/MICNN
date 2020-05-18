#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.models import *
from keras.utils import to_categorical  #独热编码
from keras.callbacks import *
from keras.layers import *
from keras.optimizers import Adam
from sklearn.metrics import *


# In[2]:


# data for stage 1 with unbalance ratio 5:1, where _1_ represents sub-integration data, while _2_ represents statistic features.
X_int_train_1_5=np.load("X_int_train_1_5.npy")
y_train_5=np.load("y_train_5.npy")


X_int_train_1_20=np.load("X_int_train_1_20.npy")
y_train_20=np.load("y_train_20.npy")


X_int_train_1=np.load("X_int_train_1_75.npy")
y_train=np.load("y_train_75.npy")

X_int_test_1=np.load("X_int_test_1.npy")
y_test=np.load("y_test.npy")


# In[3]:


# MICNN
global_input_size = (16,64,1)
global_input = Input(global_input_size)

conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(global_input)
conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
drop1 = Dropout(0.25)(pool1)

conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop1)
conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
drop2 = Dropout(0.25)(pool2)

conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop2)
conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)
drop3 = Dropout(0.25)(pool3)

flat1 = Flatten()(drop3)


dense1 = Dense(512, activation = 'relu', kernel_initializer = 'he_normal')(flat1)
drop7 = Dropout(0.5)(dense1)

dense2 = Dense(128, activation = 'relu', kernel_initializer = 'he_normal')(drop7)
drop8 = Dropout(0.5)(dense2)

dense3 = Dense(2, activation = 'softmax', kernel_initializer = 'he_normal')(drop8)



model  = Model(inputs =global_input, outputs=dense3)
model.compile(optimizer = Adam(lr = 5e-6), loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[4]:


def model_fit(model,x_train_1,y_train,x_test_1,y_test,epoch=50):
    result_all=[]
    x_train_1 = x_train_1.reshape(-1,16,64,1)
    x_test_1 = x_test_1.reshape(-1,16,64,1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    for i in range(epoch):
        result = model.fit(x_train_1, y_train,
              batch_size=150,
              epochs=1,
              shuffle=True
                      )

        y_pro = model.predict(x_test_1)
        y_pred = np.argmax(y_pro,axis=1)
        y_test_ = np.argmax(y_test,axis=1) 
        accuracy_int=sum(y_test_==y_pred)/len(y_test_)
        recall_int = recall_score(y_test_,y_pred)
        precision_int = precision_score(y_test_,y_pred)
        confusion_int = confusion_matrix(y_test_,y_pred)    
        [tn, fp], [fn, tp] = confusion_int
    #     print("FPR_int:",fp/(fp+tn))
        result_all.append([result.history['loss'][0],result.history['accuracy'][0],accuracy_int,recall_int,precision_int,fp/(fp+tn)])
    return model,result_all


# In[5]:


def model_train(model):
    model,result_1 = model_fit(model,X_int_train_1_5,y_train_5,X_int_test_1,y_test,epoch=100)
    model,result_2 = model_fit(model,X_int_train_1_20,y_train_20,X_int_test_1,y_test,epoch=100)
    model,result_3 =model_fit(model,X_int_train_1,y_train,X_int_test_1,y_test,epoch=100)
    result = np.vstack([np.array(result_1),np.array(result_2),np.array(result_3)])
    return model,result


# In[6]:


model,result = model_train(model)


# In[40]:


np.save("result_cnn.npy",result)
from keras.models import save_model
save_model(model,"MICNN.h5")


# In[ ]:




