
class ImageProcessing(object):
    """
    Will do all the work when processing frame as an image.
    Works for Evaluation or for server side evaluation (later).
    """

    def __init__(self, settings):
        self.settings = settings

    def get_crop(self, coordinates, frame):

        return 0