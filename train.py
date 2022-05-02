# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
import sys

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
size = 128

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/Main_dataset/train',
                                                 target_size=(size, size),
                                                 batch_size=4,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/Main_dataset/test',
                                            target_size=(size , size),
                                            batch_size=4,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
                                        
# Initialising the CNN
# Step 1 - Building the CNN
sz = 128
# Initializing the CNN
classifier = Sequential()
# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), strides=(1, 1), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))

# Second convolution layer and pooling
classifier.add(Convolution2D(48, (3, 3), strides=(1, 1),activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.25))

classifier.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # classifier.add(Dropout(0.25))


# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=40, activation='relu'))
classifier.add(Dense(units=27, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model
classifier.summary()
classifier.fit_generator(
        training_set, 
        epochs=30,
        validation_data=test_set)

model_json = classifier.to_json()   #weight saving
with open('/content/drive/MyDrive/Models/model-bw.json', "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('/content/drive/MyDrive/Models/model-bw.h5')
print('Weights saved')