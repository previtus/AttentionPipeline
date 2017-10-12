# Here we want to use data and model
"""
inputs: cropped images 6701+1675 images with 224x213x3 size
labels: 6701+1675 labels

dataset => [6701 train, 1675 validation]

n images * 640x640x3 ----[ CNN model ]---- n labels x 1

Possibly:
n images * 640x640x3 ----[ ResNet50 ]-[flatten]-[ custom top: dense, dropout ]-[dense sigmoid 1]-- n labels x 1

"""

import numpy as np
import os
import sys
sys.path.append("..")

from data.data_handler import *
from helpers import visualize_history
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

print "training dataset:", len(t_filenames), "image files"
print "validation dataset:", len(v_filenames), "image files"

print len(t_ids), train_labels.shape
print len(v_ids), validation_labels.shape

# load them and resize


input_shape = None
model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)



#v_filenames = v_filenames[0:21]
#validation_labels = validation_labels[0:21]
#t_filenames = t_filenames[0:21]
#train_labels = train_labels[0:21]


filename_features_train = "train_features_fullsizedata_Resnet50.npy"
filename_features_test = "val_features_fullsizedata_Resnet50.npy"
features_need_cooking = True

if features_need_cooking:
    # FULL SIZE - 32GB Mem not enough for both
    t_data = filenames_to_data(t_filenames)

    bottleneck_features_train = model.predict(t_data, batch_size=32, verbose=1)
    print "saving train_features of size", len(bottleneck_features_train), " into ", filename_features_train
    np.save(open(filename_features_train, 'w'), bottleneck_features_train)

    t_data = []


    v_data = filenames_to_data(v_filenames)

    bottleneck_features_validation = model.predict(v_data, batch_size=32, verbose=1)
    print "saving val_features of size", len(bottleneck_features_validation), " into ", filename_features_test
    np.save(open(filename_features_test, 'w'), bottleneck_features_validation)

    v_data = []