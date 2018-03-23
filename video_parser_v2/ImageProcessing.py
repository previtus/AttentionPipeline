from PIL import Image
import cv2
from keras.preprocessing.image import img_to_array

class ImageProcessing(object):
    """
    Will do all the work when processing frame as an image.
    Works for Evaluation or for server side evaluation (later).
    """

    def __init__(self, settings):
        self.settings = settings

    def get_crop(self, coordinates, img):

        #print("coordinates",coordinates)
        # coordinates come as (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + crop_size), int(h_crop[0] + crop_size))
        # it is LEFT, TOP, RIGHT, BOTTOM ?
        #crop_img = img[y:y + h, x:x + w]

        if self.settings.opencv_or_pil == 'PIL':
            cropped_img = img.crop(box=coordinates)
            if cropped_img.size[0] != 608 or cropped_img.size[1] != 608:
                print("Careful, needed to resize the crop in Evaluation->ImageProcessing. It was", cropped_img.size)
                cropped_img = cropped_img.resize((608, 608), resample=Image.ANTIALIAS)
            cropped_img.load()
        else:
            cropped_img = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
            if cropped_img.shape[0] != 608 or cropped_img.shape[1] != 608:
                print("Careful, needed to resize the crop in Evaluation->ImageProcessing. It was", cropped_img.shape)
                cropped_img = cv2.resize(cropped_img, (608, 608), interpolation=cv2.INTER_CUBIC)


        ### cropped_img = img_to_array(cropped_img) ### SHOULD BE WITH LOADER IN VIDEOCAPTURE
        return cropped_img

    def scale_image(self, image, scale):

        if self.settings.opencv_or_pil == 'PIL':
            ow, oh = image.size
            nw = ow * scale
            nh = oh * scale
            return image.resize((int(nw), int(nh)), Image.ANTIALIAS)

        else:
            oh, ow, channels = image.shape

            nw = ow * scale
            nh = oh * scale

            return cv2.resize(image, (int(nw), int(nh)), interpolation=cv2.INTER_CUBIC) # or better quality??

    # Faster alternative for image editing?
    # (unsuccesful) https://github.com/jbaiter/jpegtran-cffi
    # (alternative?) https://github.com/ethereon/lycon
    #    - according to it OpenCV should be faster

