from PIL import Image
from keras.preprocessing.image import img_to_array

class ImageProcessing(object):
    """
    Will do all the work when processing frame as an image.
    Works for Evaluation or for server side evaluation (later).
    """

    def __init__(self, settings):
        self.settings = settings

    def get_crop(self, coordinates, img):
        cropped_img = img.crop(box=coordinates)
        if cropped_img.size[0] != 608 or cropped_img.size[1] != 608:
            print("Careful, needed to resize the crop in Evaluation->ImageProcessing. It was", cropped_img.size)
            cropped_img = cropped_img.resize((608, 608), resample=Image.ANTIALIAS)
        cropped_img.load()

        ### cropped_img = img_to_array(cropped_img) ### SHOULD BE WITH LOADER IN VIDEOCAPTURE
        return cropped_img

    def scale_image(self, image, scale):

        ow, oh = image.size

        nw = ow * scale
        nh = oh * scale

        return image.resize((int(nw), int(nh)), Image.ANTIALIAS)

    # Faster alternative for image editing?
    # (unsuccesful) https://github.com/jbaiter/jpegtran-cffi
    # (alternative?) https://github.com/ethereon/lycon
    #    - according to it OpenCV should be faster

