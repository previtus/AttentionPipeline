
class CropsCoordinates(object):
    """
    Holds the functions to calculate crops coordinates when cutting up a large image and also transformation of a
    crop back into the original image coordinates.
    function: get_crops_coordinates
    """

    def __init__(self, settings):
        self.settings = settings

        settings.horizontal_splits
        settings.overlap_px
        settings.attention_horizontal_splits
        settings.attention


    def get_crops_coordinates(self, type):
        return 0
