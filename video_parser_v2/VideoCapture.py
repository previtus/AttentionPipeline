
class VideoCapture(object):
    """
    Generates individual frames from video file or from stream of connected camera. Provides image when asked for.
    """

    def __init__(self, settings):
        self.settings = settings

    def frame_generator(self):
        # this should be an iterator
        print("")
        return [0]