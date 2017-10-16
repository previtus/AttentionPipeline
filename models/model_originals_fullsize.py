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
from helpers import *
from keras.applications.resnet50 import ResNet50
from timeit import default_timer as timer
from generator_fun import generator_from_filenames

data_train, data_val = default_load(folder="images")

# LIMIT DATA
data_train = data_train#[0:21]
data_val = data_val#[0:14]

train = np.transpose(data_train)
t_filenames = train[0]
t_ids = train[1]
train_labels = np.array(train[2])

val = np.transpose(data_val)
v_filenames = val[0]
v_ids = val[1]
validation_labels = np.array(val[2])

t_filenames_and_labels = [t_filenames, train_labels]
t_filenames_and_labels = np.transpose(t_filenames_and_labels)

v_filenames_and_labels = [v_filenames, validation_labels]
v_filenames_and_labels = np.transpose(v_filenames_and_labels)

t_filenames_and_labels = [t_filenames, train_labels]
t_filenames_and_labels = np.transpose(t_filenames_and_labels)

v_filenames_and_labels = [v_filenames, validation_labels]
v_filenames_and_labels = np.transpose(v_filenames_and_labels)


print ("training dataset:", len(t_filenames), "image files")
print ("validation dataset:", len(v_filenames), "image files")

print (len(t_ids), train_labels.shape)
print (len(v_ids), validation_labels.shape)

# load them and resize


input_shape = None
model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)



#v_filenames = v_filenames[0:21]
#validation_labels = validation_labels[0:21]
#t_filenames = t_filenames[0:21]
#train_labels = train_labels[0:21]


filename_features_train = "train_features_fullsizedata_Resnet50.npy"
filename_features_val = "val_features_fullsizedata_Resnet50.npy"
features_need_cooking = False

if features_need_cooking:

    target_size = None

    # SEEMS LIKE THEY HAVE TO MAKE UP THE WHOLE SET
    #batch_size = 16
    #mod = 1
    #while mod <> 0:
    #    batch_size = batch_size - 1
    #    nbatches_train, mod = divmod(len(t_filenames_and_labels), batch_size)
    #print batch_size, "x", nbatches_train, "with mod", mod

    batch_size = 1
    nbatches_train = len(t_filenames_and_labels)

    #train_generator = generator_from_filenames(t_filenames_and_labels, batch_size, target_size)
    #bottleneck_features_train = model.predict_generator(train_generator, steps=nbatches_train, verbose=1)
    #print ""
    #print bottleneck_features_train.shape
    #np.save(open(filename_features_train, 'w'), bottleneck_features_train)

    batch_size = 1
    nbatches_val = len(v_filenames_and_labels)
    val_generator = generator_from_filenames(v_filenames_and_labels, batch_size, target_size)
    bottleneck_features_val = model.predict_generator(val_generator, steps=nbatches_val, verbose=1)
    print ""
    print bottleneck_features_val.shape
    np.save(open(filename_features_val, 'w'), bottleneck_features_val)


    """
    # FULL SIZE - 32GB Mem not enough for both
    t_data = filenames_to_data(t_filenames)
    bottleneck_features_train = model.predict(t_data, batch_size=32, verbose=1)
    print ""
    print bottleneck_features_train.shape
    #print ("saving train_features of size", len(bottleneck_features_train), " into ", filename_features_train)
    #np.save(open(filename_features_train, 'w'), bottleneck_features_train)
    np.save(open("TESTf2.npy", 'w'), bottleneck_features_train)
    """

    # time
    #    23/6701 [..............................] - ETA: 22883s


train_data = np.load(open(filename_features_train))
validation_data = np.load(open(filename_features_val))

print "training dataset:", train_data.shape, "features", train_labels.shape, "labels"
print "validation dataset:", validation_data.shape, "features", validation_labels.shape, "labels"

# build model

#### TOP MODEL
# Features - [flatten]-[ custom top: dense, dropout ]-[dense sigmoid 1]-- n labels x 1
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Model

img_features_input = Input(shape=(2, 2, 2048))
top = Flatten()(img_features_input)
top = Dense(256, activation='relu')(top)
top = Dropout(0.5)(top)
top = Dense(256, activation='relu')(top)
top = Dropout(0.5)(top)
output = Dense(1, activation='sigmoid')(top)

model = Model(inputs=img_features_input, outputs=output)
print "\n[TOP MODEL]"
param_string = short_summary(model)
print "Model widths:", param_string
print ""

# Training
epochs = 300
batch_size = 32
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

print "Time of training:", training_time, "(per sample:", (training_time/float(len(validation_labels)),"), Evaluation time:", validation_time)

history = history.history
#visualize_history(history,custom_title="Training, "+str(epochs)+" epochs, "+str(time)+"s",show=False,save=True,save_path='loss.png')
visualize_history(history,custom_title="Training, "+str(epochs)+" epochs, "+str(training_time)+"s")

info = {"epochs":epochs, "time train":training_time, "param_string":param_string}
save_history(history,"fullsize_history_B.npy",added=info)
