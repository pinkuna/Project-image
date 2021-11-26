# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:31:30 2019

@author:vinayak sable 
"""

import numpy as np       
import os                 
from random import shuffle 
import glob
import cv2
from cnn_model import get_model, get_model_keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


path = r'./FooDD'
IMG_SIZE = 256
LR = 1e-3
#Fruits_dectector-{}-{}.model
MODEL_NAME = 'Fruits_dectector-{}-{}.model'.format(LR, '5conv-basic')
no_of_fruits=8
percentage=0.3
no_of_images=200

def create_train_data(path):
    training_data = []
    labels = []
    folders=os.listdir(path)[0:no_of_fruits]
    for i in range(len(folders)):
        label = [0 for i in range(no_of_fruits)]
        label[i] = 1
        print(folders[i])
        labels.append(folders[i])
        k=0
        for j in glob.glob(path+"\\"+folders[i]+"\\*.jpg"):            
            if(k==no_of_images):
                break
            k=k+1
            img = cv2.imread(j)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            img = img/255.
            training_data.append([np.array(img),np.array(label)])
    np.save('training_{}_{}_{}'.format(no_of_fruits,no_of_images,IMG_SIZE),training_data)
    np.save('labels.npy', labels)
    shuffle(training_data)
    return training_data,folders

training_data,labels=create_train_data(path)
# training_data=np.load('training_{}_{}_{}.npy'.format(no_of_fruits,no_of_images,IMG_SIZE),allow_pickle=True)

X = np.array([i[0] for i in training_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in training_data]
Y = np.array(Y)

X, test_x, Y, test_y = train_test_split(X, Y, test_size=percentage, random_state=42)

model=get_model_keras(IMG_SIZE,no_of_fruits,LR)

history = model.fit(X, Y,epochs = 3, batch_size = 4, validation_data=(test_x, test_y))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model_save_at=os.path.join("model",MODEL_NAME)
model.save(model_save_at)
print("Model Save At",model_save_at)
