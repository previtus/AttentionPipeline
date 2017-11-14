from __future__ import print_function
# Here we want to use data and model
"""
inputs: cropped images 6701+1675 images with 224x213x3 size
labels: 6701+1675 labels

dataset => [6701 train, 1675 validation]

n images * 224x224x3 ----[ CNN model ]---- n labels x 1

Possibly:
n images * 224x224x3 ----[ ResNet50 ]-[flatten]-[ custom top: dense, dropout ]-[dense sigmoid 1]-- n labels x 1

"""

import numpy as np
import os
import sys
sys.path.append("..")

from data.data_handler import *
from helpers import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.losses import mean_squared_error

from timeit import default_timer as timer

data_train, data_val = default_load(folder="images")

train = np.transpose(data_train)
t_filenames = train[0]
t_ids = train[1]
train_labels = np.array(train[2])

val = np.transpose(data_val)
v_filenames = val[0]
v_ids = val[1]
validation_labels = np.array(val[2])

#print ("training dataset:", len(t_filenames), "image files",  len(t_ids), train_labels.shape)
#print ("validation dataset:", len(v_filenames), "image files", len(v_ids), validation_labels.shape)


#v_filenames = v_filenames[0:21]
#validation_labels = validation_labels[0:21]
#t_filenames = t_filenames[0:21]
#train_labels = train_labels[0:21]


filename_features_train = "train_features_resizeddata_Resnet50.npy"
filename_features_val = "val_features_resizeddata_Resnet50.npy"
features_need_cooking = False

if features_need_cooking:
    input_shape = None
    model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    t_data = filenames_to_data(t_filenames,target_size=(213,224)) #img_height, img_width
    v_data = filenames_to_data(v_filenames,target_size=(213,224))


    bottleneck_features_validation = model.predict(v_data, batch_size=32, verbose=1)
    print ("saving val_features of size", len(bottleneck_features_validation), " into ", filename_features_val)
    np.save(open(filename_features_val, 'w'), bottleneck_features_validation)

    bottleneck_features_train = model.predict(t_data, batch_size=32, verbose=1)
    print ("saving train_features of size", len(bottleneck_features_train), " into ", filename_features_train)
    np.save(open(filename_features_train, 'w'), bottleneck_features_train)

# load and report features

train_data = np.load(filename_features_train)
validation_data = np.load(filename_features_val)

print ("training dataset:", train_data.shape, "features", train_labels.shape, "labels")
print ("validation dataset:", validation_data.shape, "features", validation_labels.shape, "labels")

# build model

#### TOP MODEL
# Features - [flatten]-[ custom top: dense, dropout ]-[dense sigmoid 1]-- n labels x 1
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Model

img_features_input = Input(shape=(1, 1, 2048))
top = Flatten()(img_features_input)
top = Dense(64, activation='relu')(top)
top = Dropout(0.6)(top)
top = Dense(32, activation='relu')(top)
top = Dropout(0.6)(top)
output = Dense(1, activation='sigmoid')(top)

model = Model(inputs=img_features_input, outputs=output)
print ("\n[TOP MODEL]")
param_string = short_summary(model)
print ("Model widths:", param_string)
print ("")


# Training
epochs = 1000
batch_size = 28*4
model.compile(optimizer='rmsprop', loss='mean_squared_error')

start = timer()
history = model.fit(train_data, train_labels, verbose=2,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))
end = timer()
training_time = (end - start)

# Evaluate the true score:
start = timer()
validation_pred = model.predict(validation_data)
end = timer()
validation_time = (end - start)

print ("Time of training:", training_time, "(per sample:", (training_time/float(len(validation_labels)),"), Evaluation time:", validation_time))

history = history.history
#visualize_history(history,custom_title="Training, "+str(epochs)+" epochs, "+str(time)+"s",show=False,save=True,save_path='loss.png')
visualize_history(history,custom_title="Training, "+str(epochs)+" epochs, "+str(training_time)+"s")

info = {"epochs":epochs, "time train":training_time, "param_string":param_string}
save_history(history,"resized_history_1000_s.npy",added=info)
