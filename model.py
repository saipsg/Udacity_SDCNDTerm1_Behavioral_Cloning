
# coding: utf-8

# In[1]:

# loading libraries
import tensorflow as tf
import csv
import cv2
import matplotlib.image as mpimg
import numpy as np
import keras

lines=[]
images=[]

with open('./my_data/driving_log.csv') as csvfile: #Reading the csv file for the name and location of images
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines=lines[1:]
print(len(lines))
print('Done')


# In[2]:

images=[]
measurement=[]
steer=[]
#Sorting Center Camera Images 
for i in range(8566):
    source_path=lines[i][0] 
    filename=source_path.split('\\')[-1]
    #print(filename)
    current_path='./my_data/IMG/'+filename
    image=mpimg.imread(current_path)
    images.append(image)
    #Storing all the steer angles to center camera images
    measurement=float(lines[i][3]) 
    steer.append(measurement)
print('Done')
print(len(steer))



# In[4]:

# Assume a steering correction of 0.25 to correct the left and right camera images
#Adding right and left images and steering measurements to training data
measurement_left = [0.25 + x for x in steer] #Including the steering correction for left image
measurement_right = [-0.25 + x for x in steer] #Including the steering correction for right image
steer = steer + measurement_left #Add these values to the steer angle array
print(len(steer))
steer = steer + measurement_right #Add these values to the steer angle array
print(len(steer))


#Sorting all the Left camera images
for i in range(8566):
    source_path = lines[i][1]
    #     print(source_path)
    filename = source_path.split('\\')[-1]
    current_path = './my_data/IMG/' + filename
    image = mpimg.imread(current_path) 
    #Appending left camera images
    images.append(image)

#Sorting all the Right camera images
for i in range(8566):
    source_path = lines[i][2]
    #     print(source_path)
    filename = source_path.split('\\')[-1]
    current_path = './my_data/IMG/' + filename
    image = mpimg.imread(current_path) 
    #Appending right camera images
    images.append(image)
print('Done')
print(images[1].shape)


# In[6]:

#import numpy as np
X_train=np.array(images) #Converting the list into array
y_train=np.array(steer) #Converting the list into array
print(len(X_train),len(y_train))
print('Done')


# In[7]:

#Build the NVIDIA Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Convolution2D, MaxPooling2D, Lambda, ELU, Dropout, Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Nvidia - Neural Network -modified version (Added Dense (1100))
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) #Normalization of Data using a Lambda layer
model.add(Cropping2D(cropping=((50,20), (0,0)))) #Cropping the images to make it more easy for the neural network to classify
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(1100))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') # Mean square Error loss function and Adam optimizer is used
history_object=model.fit(X_train ,y_train , validation_split=0.2, shuffle=True, nb_epoch=8) # 20% images used for validation, 8 EPOCHS
model.save('modelw.h5') 




# In[ ]:



