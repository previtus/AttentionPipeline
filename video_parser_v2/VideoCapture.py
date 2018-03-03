import os, fnmatch
from processing_code.file_handling import is_non_zero_file
from PIL import Image
from keras.preprocessing.image import img_to_array

class VideoCapture(object):
    """
    Generates individual frames from video file or from stream of connected camera. Provides image when asked for.
    """

    def __init__(self, settings):
        self.settings = settings

        self.path = settings.INPUT_FRAMES
        self.init_frames_from_folder(self.path)

        self.init_settings_w_h()

    def init_frames_from_folder(self, path):
        files = sorted(os.listdir(path))
        files = [p for p in files if is_non_zero_file(path + p)]

        self.frame_files = fnmatch.filter(files, '*.jpg')
        self.annotation_files = fnmatch.filter(files, '*.xml')
        print("jpgs:", self.frame_files[0:2], "...", "xmls:", self.annotation_files[0:2], "...")

        start_frame = self.settings.startframe
        end_frame = self.settings.endframe

        if end_frame is not -1:
            frame_files = self.frame_files[start_frame:end_frame]
        else:
            frame_files = self.frame_files[start_frame:]

        self.number_of_frames = len(frame_files)

    def init_settings_w_h(self):
        first_image = self.path + self.frame_files[0]
        frame_image = Image.open(first_image)
        w, h = frame_image.size
        self.settings.set_w_h(w,h)

    def frame_generator(self):
        for i in range(self.number_of_frames):
            path = self.path + self.frame_files[i]
            image = Image.open(path)
            #### should really be here image = img_to_array(image)

            yield (path, image)