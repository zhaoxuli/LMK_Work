# -*- coding: utf-8 -*-
from keras.models import load_model
import  numpy as np
import os
import cv2
import tensorflow as tf
import keras as ks

#load model
model = load_model('./models/test_6_0.0022897404')
img_size = [32,32,3]

train_sample = np.load('./data/train.npy')[100]
# 18+4
label = train_sample[-22:]
data_in = train_sample[:-22]
data_predict = np.reshape(data_in,(1,img_size[0],img_size[1],img_size[2]))
out = model.predict(data_predict)

print ("the predict is:")
for ele in out:
    print (ele)


print ("the label is:")
print (label[:-4])
