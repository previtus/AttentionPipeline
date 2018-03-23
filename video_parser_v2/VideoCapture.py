import os, fnmatch
from processing_code.file_handling import is_non_zero_file
from PIL import Image
import cv2
from keras.preprocessing.image import img_to_array
from timeit import default_timer as timer
from threading import Thread

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

        self.last_loading_thread = None

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
        if self.settings.opencv_or_pil == 'PIL':
            frame_image = Image.open(first_image)
            w, h = frame_image.size
        else:
            frame_image = cv2.imread(first_image)
            h, w, channels = frame_image.shape

        self.settings.set_w_h(w,h)

    def frame_generator(self):
        L = self.settings.precompute_number
        L = min(self.number_of_frames, L) # careful not to go beyond

        preloaded_images = []
        for i in range(L):
            path = self.path + self.frame_files[i]
            if self.settings.opencv_or_pil == 'PIL':
                image = Image.open(path)
            else:
                image = cv2.imread(path)

            preloaded_images.append( [path, image, self.frame_files[i]] )

        f = L

        now_index = 0

        for i in range(f, self.number_of_frames+f):
            load_img_start = timer()

            frame_number = i - f

            #print("i from L to num", i, "from", L, "to", self.number_of_frames)
            # load current
            loaded = preloaded_images[now_index]
            # preload next one

            if i >= len(self.frame_files):
                preloaded_images[now_index] = None
            else:
                path = self.path + self.frame_files[i]
                if self.settings.opencv_or_pil == 'PIL':
                    image = Image.open(path)
                    image.load()
                else:
                    image = cv2.imread(path)

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

            load_img_end = timer() # time from start of this for
            OI_load_time = load_img_end - load_img_start
            self.history.report_IO_load(OI_load_time, frame_number)

            self.history.tick_loop(frame_number - 1) # time from start of last tick

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

    def frame_generator_thread_loading(self):
        # On thread
        # preloading the same, but in the loop loading different
        # just before yielding fire off a new thread

        L = self.settings.precompute_number + 1
        L = min(self.number_of_frames, L) # careful not to go beyond

        preloaded_images = []
        for i in range(L):
            path = self.path + self.frame_files[i]
            if self.settings.opencv_or_pil == 'PIL':
                image = Image.open(path)
            else:
                image = cv2.imread(path)

            preloaded_images.append( [path, image, self.frame_files[i], i] )
            print("init loading ", self.frame_files[i])
        f = L

        for i in range(f, self.number_of_frames+f):
            load_img_start = timer()

            frame_number = i - f

            #print("i from L to num", i, "from", L, "to", self.number_of_frames)
            # load current

            if self.last_loading_thread is not None:
                print("waiting for loading thread")
                self.last_loading_thread.join()

            current = preloaded_images[0]
            print("current = ", current[2])
            next_frames = preloaded_images[1:]
            for j,n in enumerate(next_frames):
                print("next[",j,"] = ", n[2])
            preloaded_images = next_frames

            i_next = i
            if (i_next < self.number_of_frames):
                print("started thread for = ", self.frame_files[i_next])
                next_frame_number = (frame_number + L)
                self.start_loading_next_image_on_thread(i_next, preloaded_images, next_frame_number)

            if self.settings.verbosity >= 2:
                print("\-------------")
                print("")
            if self.settings.verbosity >= 1:
                print("#"+str(i-f)+":", current[2], current[1].size)

            load_img_end = timer()
            OI_load_time = load_img_end - load_img_start
            self.history.report_IO_load(OI_load_time, frame_number)
            self.history.tick_loop(frame_number - 1) # time from start of last tick

            yield (current, next_frames, frame_number)

    def start_loading_next_image_on_thread(self,i_next, preloaded_images, next_frame_number):
        path_of_next = self.path + self.frame_files[i_next]
        filename_next = self.frame_files[i_next]

        t = Thread(target=self.load_on_thread, args=(path_of_next, preloaded_images, filename_next, next_frame_number))
        t.daemon = True
        t.start()

        self.last_loading_thread = t

    def load_on_thread(self, path, preloaded_images, filename, next_frame_number):

        if self.settings.opencv_or_pil == 'PIL':
            image = Image.open(path)
            image.load()
        else:
            image = cv2.imread(path)

        preloaded_images.append([path, image, filename, next_frame_number])

        print("finished thread for = ", filename, " for frame number ",next_frame_number)
