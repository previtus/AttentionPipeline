import evaluation_code.darkflow_handler as darkflow_handler
import ImageProcessing


class Evaluation(object):
    """
    Handles evaluation. Possibly with a server through Connection.
    """

    def __init__(self, settings, connection, cropscoordinates):
        self.settings = settings
        self.connection = connection
        self.cropscoordinates = cropscoordinates

        self.imageprocessing = ImageProcessing.ImageProcessing(settings)

        self.local = True
        if self.local:
            self.local_model = self.init_local()


    def evaluate(self, crops_coordinates, frame, type):
        print("Evaluation of stage",type)
        if self.local:
            return self.evaluate_local(crops_coordinates, frame, type)
        else:
            return 0


    # Assuming there is no server via Connection, lets evaluate it here on local machine
    def init_local(self):

        local_model = darkflow_handler.load_model()

        return local_model

    def evaluate_local(self, crops_coordinates, frame, type):
        frame_path = frame[0]
        frame_image_original = frame[1]

        if type == 'attention':
            frame_image = self.imageprocessing.scale_image(frame_image_original, self.cropscoordinates.scale_ratio_of_attention_crop)
            print("attention scaled size to", frame_image.size)

        elif type == 'evaluation':
            frame_image = self.imageprocessing.scale_image(frame_image_original, self.cropscoordinates.scale_ratio_of_evaluation_crop)
            print("evaluation scaled size to", frame_image.size)

        ids_of_crops = []
        crops = []
        for coordinates in crops_coordinates:
            coordinates_id = coordinates[0]
            coordinates_area = coordinates[1]

            crop = self.imageprocessing.get_crop(coordinates_area, frame_image)
            crops.append(crop)
            ids_of_crops.append(coordinates_id)

        evaluation = darkflow_handler.run_on_images(crops, self.local_model)
        evaluation = [[ids_of_crops[i], eval] for i,eval in enumerate(evaluation)]

        evaluation = self.filter_evaluations(evaluation)

        # Returns evaluation in format:
        # array of crops in order by coordinates_id
        # each holds id and array of dictionaries for each bbox {} keys label, confidence, topleft, bottomright
        return evaluation

    def filter_evaluations(self, evaluation):

        # confidence filtered by the model already
        # see load_model() in darkflow_handler

        evaluation = self.filter_label(evaluation, 'person')

        return evaluation

    def filter_label(self, evaluation, label):
        filtered = []
        for id, evaluation_in_crop in evaluation:
            if len(evaluation_in_crop)>0:
                replace = []
                for detected_object in evaluation_in_crop:
                    if detected_object["label"] == label:
                        replace.append(detected_object)

                filtered.append([id,replace])
                # whats the most efficient way of filtering items in list?
        return filtered