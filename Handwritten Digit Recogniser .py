#!/usr/bin/env python
# coding: utf-8

# In[108]:


#LOAD THE NECESSARY LIBRARIES
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D
from keras.utils import to_categorical


# # INSTALLATION OF MNIST DATASET

# In[109]:


pip install mnist


# # LOADING THE DATASET

# In[110]:


#Load the data set
train_images = mnist.train_images()

train_labels = mnist.train_labels()

test_images = mnist.test_images()

test_labels = mnist.test_labels()


# In[111]:


# Normalize the images
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5


# In[112]:


train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))
print(train_images.shape)
print(test_images.shape)


# In[ ]:


train_images.shape


# # BUILDING THE MODEL

# In[7]:


# Building the model
# 3 layers , 2 layers with 64 neurons and the relu function
# 1 layer with 10 neurons and softmax function
model = Sequential()
model.add (Dense(64,activation = 'relu', input_dim=784))
model.add(Dense(64,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))


# In[8]:


# The loss function measures how well the model did on training and to improve using optimizer
model.compile(
    optimizer = 'adam',
     loss = 'categorical_crossentropy',#classes that are greater than 2
     metrics = ['accuracy']


)


# In[9]:


# training the model
model.fit(
    train_images,
      to_categorical(train_labels), # Ex 2 it expects [0,0,1,0,0,0,0,0,0,0]
      epochs = 5,
      batch_size=32
)


# In[10]:


# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)


# In[11]:


# model.save_weights('model.h5')


# In[12]:


# predict the first 5 test images of our model
predictions = model.predict(test_images[:5])
# print our models prediction
print(np.argmax(predictions, axis = 1))
print(test_labels[:5])


# In[13]:


for i in range(0,5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


# In[14]:


import cv2

image = cv2.imread('C:\\Users\\JAYALAKSHMI\\Downloads\\one_image.jpeg')
grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
preprocessed_digits = []
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    
    # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
    
    # Cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y+h, x:x+w]
    
     # Resizing that digit to (18, 18)
    resized_digit = cv2.resize(digit, (18,18))
    
    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
    
    # Adding the preprocessed digit to the list of preprocessed digits
    preprocessed_digits.append(padded_digit)
print("\n\n\n----------------Contoured Image--------------------")
plt.imshow(image, cmap="gray")
plt.show()

inp = np.array(preprocessed_digits)


# In[15]:


for digit in preprocessed_digits:
    prediction = model.predict(digit.reshape(1,784))
    print("\n\n\n--------------\n\n\n")
    print("======PREDICTION====== \n\n\n")
    plt.imshow(digit.reshape(28,28), cmap="gray")
    plt.show()
    print("\n\n Final output:{}".format(np.argmax(prediction)))
    print("\n Prediction(Softmax) from the neural network:\n\n {}".format(prediction))
    hard_maxed_prediction = np.zeros(prediction.shape)
    hard_maxed_prediction[0][np.argmax(prediction)] = 1
    print("\n\n Hard_maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
    print("\n\n\n------------------\n\n\n")


# In[ ]:




