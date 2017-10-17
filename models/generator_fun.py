from __future__ import print_function
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# adapted from https://techblog.appnexus.com/a-keras-multithreaded-dataframe-generator-for-millions-of-image-files-84d3027f6f43
# however we need only simpler version
def generator_from_filenames(data, batch_size, resize):
    """
    data = [ [filename1, y1],[filename2, y2], ... ]

    Generator that yields (X, Y) from list of (Path, Y)
    """

    nbatches, n_skipped_per_epoch = divmod(len(data), batch_size)
    count = 1
    epoch = 0

    while 1:

        epoch += 1
        i, j = 0, batch_size
        mini_batches_completed = 0

        for _ in range(nbatches):

            sub = data[i:j]

            try:
                X = np.array([img_to_array(load_img(item[0], target_size=resize)) for item in sub])
                # more possible preprocessing here
                Y = [item[1] for item in sub]
                Y = np.array(Y).astype(float)

                mini_batches_completed += 1
                yield X, Y

            except IOError as err:
                print ("IOERROR", err)
                count -= 1

            i = j
            j += batch_size
            count += 1