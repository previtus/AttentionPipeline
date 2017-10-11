# Here we want to use data and model
"""
inputs: cropped images 25128 images with 224x224x3 size
labels: 25128 labels

25128 dataset => [20102 train, 5026 validation]

n images * 224x224x3 ----[ CNN model ]---- n labels x 1

Possibly:
n images * 224x224x3 ----[ ResNet50 ]-[flatten]-[ custom top: dense, dropout ]-[dense sigmoid 1]-- n labels x 1

"""

import numpy as np
import os
import sys
sys.path.append("..")

from data.data_handler import *
from helpers import visualize_history
from keras.applications.resnet50 import ResNet50

data_train, data_val = default_load()

train = np.transpose(data_train)
t_filenames = train[0]
train_labels = np.array(train[1])

val = np.transpose(data_val)
v_filenames = val[0]
validation_labels = np.array(val[1])

print "training dataset:", len(t_filenames), "image files"
print "validation dataset:", len(v_filenames), "image files"

features_need_cooking = False

# small version
#v_filenames = v_filenames[0:20]
#validation_labels = validation_labels[0:20]
#t_filenames = t_filenames[0:20]
#train_labels = train_labels[0:20]

#### BASE MODEL
# n images * 224x224x3 ----[ ResNet50 ]- Features

input_shape = None
model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
filename_features_train = "train_features_cropdata.npy"
filename_features_test = "val_features_cropdata.npy"

if features_need_cooking:

    t_data = filenames_to_data(t_filenames)
    v_data = filenames_to_data(v_filenames)
    # 25.7GB/32GB MEM
    # 1962s + 7846s

    #t_data: (20102, 224, 224, 3) images
    #v_data: (5026, 224, 224, 3) images

    #train_generator = getImageGenerator(t_filenames, t_scores)
    #val_generator = getImageGenerator(v_filenames, v_scores)

    num_train = len(train_labels)
    num_val = len(validation_labels)

    #bottleneck_features_validation = model.predict_generator(val_generator, steps=num_val, verbose=1)
    bottleneck_features_validation = model.predict(v_data, batch_size=32, verbose=1)

    print "saving val_features of size", len(bottleneck_features_validation), " into ", filename_features_test
    np.save(open(filename_features_test, 'w'), bottleneck_features_validation)

    #bottleneck_features_train = model.predict_generator(train_generator, steps=num_train, verbose=1)
    bottleneck_features_train = model.predict(t_data, batch_size=32, verbose=1)

    print "saving train_features of size", len(bottleneck_features_train), " into ", filename_features_train
    np.save(open(filename_features_train, 'w'), bottleneck_features_train)

# JUST LOAD FEATURES

train_data = np.load(open(filename_features_train))
validation_data = np.load(open(filename_features_test))

print "training dataset:", train_data.shape, "features", train_labels.shape, "labels"
print "validation dataset:", validation_data.shape, "features", validation_labels.shape, "labels"
#training dataset: (20102, 1, 1, 2048) features (20102,) labels
#validation dataset: (5026, 1, 1, 2048) features (5026,) labels


#### TOP MODEL
# Features - [flatten]-[ custom top: dense, dropout ]-[dense sigmoid 1]-- n labels x 1
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Input, concatenate, GlobalAveragePooling2D


img_features_input = Input(shape=(1, 1, 2048))
top = Flatten()(img_features_input)
top = Dense(256, activation='relu')(top)
top = Dropout(0.5)(top)
top = Dense(256, activation='relu')(top)
top = Dropout(0.5)(top)
output = Dense(1, activation='sigmoid')(top)

model = Model(inputs=img_features_input, outputs=output)

#model.summary()
epochs = 100
batch_size = 256

model.compile(optimizer='rmsprop', loss='mean_squared_error')
history = model.fit(train_data, train_labels, verbose=2,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))

history = history.history
visualize_history(history)