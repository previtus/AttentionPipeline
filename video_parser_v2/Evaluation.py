import evaluation_code.darkflow_handler as darkflow_handler
import ImageProcessing


class Evaluation(object):
    """
    Handles evaluation. Possibly with a server through Connection.
    """

    def __init__(self, settings, connection, cropscoordinates, history):
        self.settings = settings
        self.connection = connection
        self.cropscoordinates = cropscoordinates
        self.history = history

        self.imageprocessing = ImageProcessing.ImageProcessing(settings)

        self.local = not (self.settings.client_server)
        if self.local:
            if self.settings.verbosity >= 2:
                print("Evaluation init, local model of Darkflow")
            self.local_model = self.init_local()


    def evaluate(self, crops_coordinates, frame, type):
        frame_path = frame[0]
        frame_image_original = frame[1]

        if type == 'attention':
            frame_image = self.imageprocessing.scale_image(frame_image_original, self.cropscoordinates.scale_ratio_of_attention_crop)

        elif type == 'evaluation':
            frame_image = self.imageprocessing.scale_image(frame_image_original, self.cropscoordinates.scale_ratio_of_evaluation_crop)

        ids_of_crops = []
        crops = []
        for coordinates in crops_coordinates:
            coordinates_id = coordinates[0]
            coordinates_area = coordinates[1]

            crop = self.imageprocessing.get_crop(coordinates_area, frame_image)
            crops.append(crop)
            ids_of_crops.append(coordinates_id)


        if self.local:
            # should we even have this?
            from keras.preprocessing.image import img_to_array
            crops = [img_to_array(crop) for crop in crops]

            evaluation = self.evaluate_local(crops, ids_of_crops)
        else:
            evaluation = self.evaluate_on_server(crops, ids_of_crops)

        evaluation = self.filter_evaluations(evaluation)

        if self.settings.verbosity >= 2:
            counts = [len(in_one_crop[1]) for in_one_crop in evaluation]
            print("Evaluation (server) of stage `"+type+"`, bboxes in crops", counts)

        # Returns evaluation in format:
        # array of crops in order by coordinates_id
        # each holds id and array of dictionaries for each bbox {} keys label, confidence, topleft, bottomright
        return evaluation

    # Assuming there is no server via Connection, lets evaluate it here on local machine
    def init_local(self):

        local_model = darkflow_handler.load_model()

        return local_model

    def evaluate_local(self, crops, ids_of_crops):
        evaluation = darkflow_handler.run_on_images(crops, self.local_model)
        evaluation = [[ids_of_crops[i], eval] for i,eval in enumerate(evaluation)]

        return evaluation

    def evaluate_on_server(self, crops, ids_of_crops):
        evaluation = self.connection.evaluate_crops_on_server(crops, ids_of_crops)
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