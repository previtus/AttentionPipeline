from data.data_handler import *

data_train, data_val = default_load(folder="images")
train = np.transpose(data_train)
t_filenames = train[0]
t_ids = train[1]
train_labels = np.array(train[2])

val = np.transpose(data_val)
v_filenames = val[0]
v_ids = val[1]
validation_labels = np.array(val[2])

print "FULL IMAGES"
print "training dataset:", len(t_filenames), "image files", len(t_ids), train_labels.shape
print "validation dataset:", len(v_filenames), "image files", len(v_ids), validation_labels.shape
print ""

data_train, data_val = default_load()
train = np.transpose(data_train)
t_filenames = train[0]
t_ids = train[1]
train_labels = np.array(train[2])

val = np.transpose(data_val)
v_filenames = val[0]
v_ids = val[1]
validation_labels = np.array(val[2])

print "CROP IMAGES"
print "training dataset:", len(t_filenames), "image files", len(t_ids), train_labels.shape
print "validation dataset:", len(v_filenames), "image files", len(v_ids), validation_labels.shape
