import evaluation_code.darkflow_handler as darkflow_handler
import ImageProcessing

class Evaluation(object):
    """
    Handles evaluation. Possibly with a server through Connection.
    """

    def __init__(self, settings, connection):
        self.settings = settings
        self.connection = connection

        self.imageprocessing = ImageProcessing.ImageProcessing(settings)

        self.local = True
        if self.local:
            self.local_model = self.init_local()


    def evaluate(self, crops_coordinates, frame):
        if self.local:
            return self.evaluate_local(crops_coordinates, frame)
        else:
            return 0


    # Assuming there is no server via Connection, lets evaluate it here on local machine
    def init_local(self):

        local_model = darkflow_handler.load_model()

        return local_model

    def evaluate_local(self, crops_coordinates, frame):
        for coordinates in crops_coordinates:
            crop = self.imageprocessing.get_crop(coordinates, frame)

            evaluation = darkflow_handler.run_on_image(crop, self.local_model)

            print(evaluation)
