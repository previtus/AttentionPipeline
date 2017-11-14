from keras.preprocessing.image import load_img, img_to_array
import numpy as np

"""
def generator_images_scores(order, image_paths, scores, resize=None):
    '''
    Get generator of images
    :param order: prearanged order (1,2,3...) or (2,55,1,980, ...)
    :param image_paths: paths to images, these are kept in memory while the big 640x640x3 image data is not
    :param scores: score to be associated with returned image
    :param resize: parameter to resize loaded images on the fly
    :return: generator, which yields (image, score)
    '''
    while True:
        for index in order:
            img_path = image_paths[index]

            pil_img = load_img(img_path, target_size=resize)
            arr = img_to_array(pil_img, 'channels_last')
            image = np.array(arr)

            score = scores[index]
            yield (image, score)


def getImageGenerator(x_paths,y, resize=None):
    size = len(y)
    order = range(size)
    image_generator = generator_images_scores(order, image_paths=x_paths, scores=y, resize=resize)
    return image_generator


def predict_features(model, train_generator, val_generator, num_train, num_val, filename_features_train, filename_features_test):
    bottleneck_features_train = model.predict_generator(train_generator, steps=num_train,verbose=1)
    print "saving train_features of size", len(bottleneck_features_train), " into ",filename_features_train
    np.save(open(filename_features_train, 'w'), bottleneck_features_train)

    bottleneck_features_validation = model.predict_generator(val_generator, steps=num_val,verbose=1)
    print "saving val_features of size", len(bottleneck_features_validation), " into ",filename_features_test
    np.save(open(filename_features_test, 'w'), bottleneck_features_validation)
"""