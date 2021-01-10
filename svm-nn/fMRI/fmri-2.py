# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:42:31 2019

@author: 
"""

import scipy.io
import os
import numpy as np
path = "../assignment-4-Neuro-data/Neuro_dataset/"
mat_files = []
files = os.listdir(path)

path1 = path+files[0]+"/"
path2 = path+files[1]+"/"


Alz_sub = []
Alz_tc_rest_aal = []
Alz_tc_rest_power = []
for i in os.listdir(path1):
    alz = scipy.io.loadmat(path1+i)
    sub_id = alz['sub_id']
    tc_rest_aal = alz['tc_rest_aal']
    tc_rest_power = alz['tc_rest_power']
    Alz_sub.append(sub_id)
    Alz_tc_rest_aal.append(tc_rest_aal)
    Alz_tc_rest_power.append(tc_rest_power)

Alz_sub = np.array(Alz_sub)
Alz_tc_rest_aal = np.array(Alz_tc_rest_aal)
Alz_tc_rest_power = np.array(Alz_tc_rest_power)

N_sub = []
N_tc_rest_aal = []
N_tc_rest_power = []
for i in os.listdir(path2):
    n = scipy.io.loadmat(path2+i)
    sub_id = n['sub_id']
    tc_rest_aal = n['tc_rest_aal']
    tc_rest_power = n['tc_rest_power']
    N_sub.append(sub_id)
    N_tc_rest_aal.append(tc_rest_aal)
    N_tc_rest_power.append(tc_rest_power)

N_sub = np.array(N_sub)
N_tc_rest_aal = np.array(N_tc_rest_aal)
N_tc_rest_power = np.array(N_tc_rest_power)


X = np.concatenate((Alz_tc_rest_power,N_tc_rest_power))
Y = np.concatenate((np.zeros((34,1)),np.ones((47,1))))
X = X.reshape(np.shape(X)[0],np.shape(X)[1]*np.shape(X)[2])
from keras.utils import to_categorical
y_binary = to_categorical(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_binary, test_size=0.15, random_state=42)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

model = Sequential()
model.add(Dense(16, activation='relu',input_dim=(36960)))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

model.fit(X_train, y_train,batch_size=8,epochs=20,verbose=1,validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy is : ")
print(score[1]*100)
