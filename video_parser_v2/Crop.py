
class Crop(object):
    """
    One crop is defined by position id (which CropsCoordinates used to reproject it back to original image).
    Boolean if its active (AttentionModel chooses that). Coordinates which specify 4 values of where it is reaching.
    It can (but maybe shouldn't) contain the image information.
    It can contain bounding boxes found inside it (use CropsCoordinates to get it back).

    Attributes:
        a: ...
    """

    def __init__(self, position_id, coordinates, active=True):
        self.position_id = position_id
        self.coordinates = coordinates
        self.active = active

        #self.image = None
        self.bboxes = None

    def set_bboxes(self, bboxes):
        self.bboxes = bboxes


