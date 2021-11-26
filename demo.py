from calorie import calories
from cnn_model import get_model, get_model_keras
import os  
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf

IMG_SIZE = 256
LR = 1e-3
no_of_fruits=8

MODEL_NAME = 'Fruits_dectector-{}-{}.model'.format(LR, '5conv-basic')

model_save_at=os.path.join("model",MODEL_NAME)
# model = get_model(IMG_SIZE,no_of_fruits,LR)
# model = get_model_keras(IMG_SIZE,no_of_fruits,LR)
model_keras = keras.models.load_model(model_save_at)
# model.load(model_save_at)
labels=list(np.load('labels.npy'))

test_data='orenge.jpg'
img=cv2.imread(test_data)

img1=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
img1 = np.expand_dims(img1, axis=0)
model_out=model_keras.predict([img1])
result=np.argmax(model_out)
name=labels[result]

#print(img.shape)#(h,w)
d = 1024 / img.shape[1]
dim = (1024, int(img.shape[0] * d))
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA) #cv2(w, h)
cal=round(calories(result+1,img),2)

import matplotlib.pyplot as plt
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title('{}({}kcal)'.format(name,cal))
plt.axis('off')
plt.show()