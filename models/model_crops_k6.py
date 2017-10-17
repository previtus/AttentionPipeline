from __future__ import print_function
# Here we want to use data and model
"""
inputs: cropped images 25128 images with 224x224x3 size
labels: 25128 labels

25128 dataset => [20103 train, 5025 validation]

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
from keras.applications.resnet50 import ResNet50

from timeit import default_timer as timer
from generator_fun import generator_from_filenames

data_train, data_val = default_load(folder="crops_50proc_6clusters")

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

# small version
#v_filenames = v_filenames[0:21]
#validation_labels = validation_labels[0:21]
#t_filenames = t_filenames[0:21]
#train_labels = train_labels[0:21]

#### BASE MODEL
# n images * 224x224x3 ----[ ResNet50 ]- Features
# Feature size by model:
# ResNet50 1, 1, 2048
# VGG16 7, 7, 512
# VGG19 7, 7, 512
# InceptionV3 5, 5, 2048
# Xception 7, 7, 2048

filename_features_train = "train_features_cropdata_Resnet_6clusters.npy"
filename_features_val = "val_features_cropdata_Resnet_6clusters.npy"
features_need_cooking = False


if features_need_cooking:

    input_shape = None
    model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    target_size = None
    batch_size = 1
    nbatches_train = len(t_filenames_and_labels)

    train_generator = generator_from_filenames(t_filenames_and_labels, batch_size, target_size)
    bottleneck_features_train = model.predict_generator(train_generator, steps=nbatches_train, verbose=1)
    print ("")
    print (bottleneck_features_train.shape)
    np.save(open(filename_features_train, 'w'), bottleneck_features_train)

    batch_size = 1
    nbatches_val = len(v_filenames_and_labels)
    val_generator = generator_from_filenames(v_filenames_and_labels, batch_size, target_size)
    bottleneck_features_val = model.predict_generator(val_generator, steps=nbatches_val, verbose=1)
    print ("")
    print (bottleneck_features_val.shape)
    np.save(open(filename_features_val, 'w'), bottleneck_features_val)


# JUST LOAD FEATURES

train_data = np.load(open(filename_features_train))
validation_data = np.load(open(filename_features_val))

print ("training dataset:", train_data.shape, "features", train_labels.shape, "labels")
print ("validation dataset:", validation_data.shape, "features", validation_labels.shape, "labels")
#training dataset: (20102, 1, 1, 2048) features (20102,) labels
#validation dataset: (5026, 1, 1, 2048) features (5026,) labels


#### TOP MODEL
# Features - [flatten]-[ custom top: dense, dropout ]-[dense sigmoid 1]-- n labels x 1
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Input, concatenate, GlobalAveragePooling2D


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

epochs = 1000
batch_size = 108 #div by 6

from keras import backend as K
import tensorflow as tf

def grouped_mse(k=3):
    def clustered_mse(y_true, y_pred):
        group_by = tf.constant(k)
        real_size = tf.size(y_pred)

        remainder = tf.truncatemod(real_size, group_by)
        #remainder = K.print_tensor(remainder, message="remainder is: ")

        # ignore the rest
        y_true = y_true[0:real_size - remainder]
        y_pred = y_pred[0:real_size - remainder]

        real_size = tf.size(y_pred)# + 0*remainder
        #real_size = K.print_tensor(real_size, message="real_size is: ")
        n = real_size / group_by
        #n = K.print_tensor(n, message="n is: ")


        idx = tf.range(n)
        idx = tf.reshape(idx, [-1, 1])  # Convert to a len(yp) x 1 matrix.
        idx = tf.tile(idx, [1, group_by])  # Create multiple columns.
        idx = tf.reshape(idx, [-1])  # Convert back to a vector.


        y_pred_byK = tf.segment_mean(y_pred, idx) #segment_ids should be the same size as dimension 0 of input.
        y_true_byK = tf.segment_mean(y_true, idx) #segment_ids should be the same size as dimension 0 of input
        tmp = K.mean(K.square(y_pred_byK - y_true_byK), axis=-1)
        return tmp
        """
        # whoops how can it be n/3 sized loss?
        a = tf.size(y_pred_byK)
        b = tf.size(y_true_byK)

        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)

        a = K.print_tensor(a, message="a is: ")
        b = K.print_tensor(b, message="b is: ")

        tmp = y_pred_byK - y_true_byK
        tmp = K.print_tensor(tmp, message="tmp1 is: ")
        tmp = K.square(tmp)
        tmp = K.print_tensor(tmp, message="tmp2 is: ")
        tmp = K.mean(tmp, axis=-1)
        tmp = K.print_tensor(tmp, message="tmp3 is: ")

        tmp = tf.scalar_mul(1+0*a+0*b,tmp)
        """
        return tmp
    return clustered_mse

k=6
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=[grouped_mse(k)])

# Incompatible shapes: [7] vs. [21] // n - n/3
#model.compile(optimizer='rmsprop', loss=grouped_mse(k), metrics=['mean_squared_error'])

#model.compile(optimizer='rmsprop', loss='mean_squared_error')

# PS: shuffle=True applies for training data, but not validation!
start = timer()
history = model.fit(train_data, train_labels, verbose=2,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))
end = timer()
training_time = (end - start)


history = history.history

print (history)
#visualize_history(history,custom_title="Training, "+str(epochs)+" epochs, "+str(time)+"s",show=False,save=True,save_path='loss.png')
visualize_history(history,custom_title="Training, "+str(epochs)+" epochs, "+str(training_time)+"s",
                  show_also='clustered_mse')

"""
# Evaluate the true score:
start = timer()
validation_pred = model.predict(validation_data)
end = timer()
time = (end - start)

print ("time of eval:", time, ", per sample:", (time/float(len(validation_labels))))


print (validation_pred.shape, validation_labels.shape)
print (v_ids[0:7])

# Get whichever k we chose
#k = 0
#while v_ids[k] == v_ids[0]:
#    k += 1
# spoiler, it was 3

original_images = int( len(validation_labels) / k )

y_true = []
y_pred = []
for i in range(original_images):
    #v_ids[i*k:i*k+k]
    ground_truth = validation_labels[i*k]
    predicted_values = validation_pred[i*k:i*k+k]
    prediction = np.mean(predicted_values)
    y_true.append(ground_truth)
    y_pred.append(prediction)
y_true = np.array(y_true).astype(float)
y_pred = np.array(y_pred).astype(float)

print (y_true.shape,y_pred.shape)

mse1 = np.mean(np.square(y_true - y_pred))

from sklearn.metrics import mean_squared_error
mse2 = mean_squared_error(y_true, y_pred)
print (mse1, mse2)
"""

info = {"epochs":epochs, "time train":training_time, "param_string":param_string, "use_param":"clustered_mse"}
save_history(history,"crops_history_6k_1000_B.npy",added=info)
