import os, fnmatch
from processing_code.file_handling import is_non_zero_file
from PIL import Image
from keras.preprocessing.image import img_to_array
from timeit import default_timer as timer

class VideoCapture(object):
    """
    Generates individual frames from video file or from stream of connected camera. Provides image when asked for.
    """

    def __init__(self, settings, history):
        self.settings = settings
        self.history = history

        self.path = settings.INPUT_FRAMES
        self.init_frames_from_folder(self.path)

        self.init_settings_w_h()

    def init_frames_from_folder(self, path):
        files = sorted(os.listdir(path))
        files = [p for p in files if is_non_zero_file(path + p)]

        self.frame_files = fnmatch.filter(files, '*.jpg')
        self.annotation_files = fnmatch.filter(files, '*.xml')
        if self.settings.verbosity >= 2:
            print("VideoCapture init, from folder: jpgs:", self.frame_files[0:2], "...", "xmls:", self.annotation_files[0:2], "...")

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
        L = self.settings.precompute_number
        L = min(self.number_of_frames, L) # careful not to go beyond

        preloaded_images = []
        for i in range(L):
            path = self.path + self.frame_files[i]
            image = Image.open(path)

            preloaded_images.append( [path, image, self.frame_files[i]] )

        f = L

        now_index = 0

        for i in range(f, self.number_of_frames+f):
            frame_number = i - f

            #print("i from L to num", i, "from", L, "to", self.number_of_frames)
            # load current
            loaded = preloaded_images[now_index]
            # preload next one

            if i >= len(self.frame_files):
                preloaded_images[now_index] = None
            else:
                path = self.path + self.frame_files[i]
                image = Image.open(path)

                preloaded_images[now_index] = [path, image, self.frame_files[i]]

            now_index = (now_index + 1) % L

            # return current + next images
            next_frames = []
            for j in range(L):
                #print(j,"=",preloaded_images[(now_index+j) % L])
                obj = preloaded_images[(now_index+j) % L]
                if obj is not None:
                    next_frame_number = (frame_number+j+1)
                    obj.append(next_frame_number)
                    next_frames.append( obj )

            if self.settings.verbosity >= 2:
                print("\-------------")
                print("")
            if self.settings.verbosity >= 1:
                print("#"+str(i-f)+":", loaded[2], loaded[1].size)

            self.history.tick_loop(frame_number - 1)


            yield (loaded, next_frames, frame_number)

    """            
    def frame_generator(self):
        for i in range(self.number_of_frames):
            path = self.path + self.frame_files[i]
            image = Image.open(path)
            #### should really be here image = img_to_array(image)

            if self.settings.verbosity >= 2:
                print("\-------------")
                print("")
            if self.settings.verbosity >= 1:
                print("#"+str(i)+":", self.frame_files[i], image.size)

            self.history.tick_loop()

            yield (path, image, self.frame_files[i])
    """