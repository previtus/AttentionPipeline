
class Renderer(object):
    """
    Draw final image with bboxes to a screen or file.
    """

    def __init__(self, settings):
        self.settings = settings

    def render(self, final_evaluation, frame):
        return 0