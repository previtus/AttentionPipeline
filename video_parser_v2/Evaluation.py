import evaluation_code.darkflow_handler as darkflow_handler
import ImageProcessing
from timeit import default_timer as timer
from threading import Thread
import numpy as np

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

        if self.settings.precompute_attention_evaluation:
            # initialize hashmaps / dictionaries
            self.precomputations_started = {} # <frame_number> => thread id
            self.precomputations_finished = {} # <frame_number> => results
            #self.precomputing_threads = []


    def evaluate_attention_with_precomputing(self, frame_number, crops_coordinates, frame, type, next_frames):
        if not self.settings.precompute_attention_evaluation or \
                (len(next_frames) < 1 and not (frame_number in self.precomputations_started.keys() or frame_number in self.precomputations_finished.keys()))\
                or self.connection.number_of_server_machines <= 1:
            # if the feature is off, evaluate normally

            return self.evaluate(crops_coordinates, frame, type, frame_number)
        else:
            result = None
            # We know that we are precomputing attention
            # - look if we didn't start computing frame <frame_number> already -> wait for the evaluation to finish
            # - look if we didn't complete computing for frame <frame_number> already -> get data
            # - remove <frame_number> from both lists
            # - spawn new evaluation, the next one not yet started or finished from next_frames
            #
            # keep hashmaps
            # precomputations_started: <frame_number> => thread
            # precomputations_finished = <frame_number> => results

            start = timer()

            # 1.) resolve the current frame
            if frame_number in self.precomputations_finished.keys():
                result = self.precomputations_finished[frame_number]

                if self.settings.verbosity >= 3:
                    print("[precomp] Using precomputed",frame_number,".")

                del self.precomputations_finished[frame_number]

                # clean from the other hashmap too
                if frame_number in self.precomputations_started.keys():
                    del self.precomputations_started[frame_number]

            elif frame_number in self.precomputations_started.keys():
                thread = self.precomputations_started[frame_number]

                if self.settings.verbosity >= 3:
                    print("[precomp] Waiting for precomputation for",frame_number," on thread", thread)

                # wait for it to finish
                thread.join()

                result = self.precomputations_finished[frame_number]

                if frame_number in self.precomputations_started.keys():
                    del self.precomputations_started[frame_number]

                # clean from the other hashmap too
                if frame_number in self.precomputations_finished.keys():
                    del self.precomputations_finished[frame_number]

            else:
                # this frame didn't start yet, it must be the first one, evaluate it the old way...
                if self.settings.verbosity >= 3:
                    print("[precomp] Frame", frame_number, " had to be started on its own")
                result = self.evaluate(crops_coordinates, frame, type, frame_number)

            end = timer()
            waited_for_attention_in_total = end - start
            self.history.report_evaluation_attention_waiting(waited_for_attention_in_total, frame_number)

            # 2.) start of the next frame (s)
            self.start_precomputation(next_frames, crops_coordinates, type)

            return result

    def start_precomputation(self, next_frames, shared_crops_coordinates, shared_type):
        if self.settings.verbosity >= 3:
            print("[precomp] Starting threads")
        for i in range(0, len(next_frames)):
            frame = next_frames[i]
            frame_number = frame[3]
            if frame_number in self.precomputations_started:
                continue
            # lenght of next_frames depends on settings.precompute_number and what is remaining

            if self.settings.verbosity >= 3:
                print("[precomp] Starting thread for <frame_number>", frame_number)

            t = Thread(target=self.evaluate_precompute, args=(shared_crops_coordinates, frame, shared_type, frame_number))
            t.daemon = True
            t.start()
            #self.precomputing_threads.append(t)

            self.precomputations_started[frame_number] = t

    def evaluate_precompute(self, crops_coordinates, frame, type, frame_number):
        # Thread function called to precompute for next frames

        result = self.evaluate(crops_coordinates, frame, type, frame_number)

        self.precomputations_finished[frame_number] = result
        del self.precomputations_started[frame_number]


    def evaluate(self, crops_coordinates, frame, type, frame_number):
        time_start = timer()
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
        t_after_cutting = timer()
        IO_time_to_cut_crops = t_after_cutting - time_start
        #print("scaling and cutting image into crops took ", IO_time_to_cut_crops, "for type=",type,"frame_number=",frame_number)
        self.history.report_IO_EVAL_cut_evaluation(type, IO_time_to_cut_crops, frame_number)


        if self.local:
            # should we even have this?
            from keras.preprocessing.image import img_to_array
            crops = [img_to_array(crop) for crop in crops]

            evaluation = self.evaluate_local(crops, ids_of_crops)
        else:
            evaluation = self.evaluate_on_server(crops, ids_of_crops, type, frame_number)

        evaluation = self.filter_evaluations(evaluation)

        if self.settings.verbosity >= 2:
            counts = [len(in_one_crop[1]) for in_one_crop in evaluation]
            where = 'local'
            if self.settings.client_server: where = 'server'
            print("Evaluation ("+where+") of stage `"+type+"`, bboxes in crops", counts)

        time_whole_eval = timer() - time_start
        self.history.report_evaluation_whole_function(type, time_whole_eval, frame_number)

        # Returns evaluation in format:
        # array of crops in order by coordinates_id
        # each holds id and array of dictionaries for each bbox {} keys label, confidence, topleft, bottomright
        return evaluation

    # Assuming there is no server via Connection, lets evaluate it here on local machine
    def init_local(self):

        local_model = darkflow_handler.load_model(0)

        return local_model

    def evaluate_local(self, crops, ids_of_crops):
        evaluation = darkflow_handler.run_on_images(crops, self.local_model)
        evaluation = [[ids_of_crops[i], eval] for i,eval in enumerate(evaluation)]

        return evaluation

    def evaluate_on_server(self, crops, ids_of_crops, type, frame_number):
        evaluation, times_encode, times_eval, times_decode, times_transfer = self.connection.evaluate_crops_on_server(crops, ids_of_crops, type)
        # if its final evaluation, save individual times per servers
        if type == 'evaluation':
            self.history.report_evaluation_per_individual_worker(times_encode, times_eval, times_decode, times_transfer, type, frame_number)

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