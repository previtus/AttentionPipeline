import requests
from multiprocessing.pool import ThreadPool
from io import BytesIO
import math
from threading import Thread
import numpy as np
from random import shuffle

from PIL import Image
import cv2
from timeit import default_timer as timer

class Connection(object):
    """
    Holds connection to server(s), handles sending and receiving to and from the right one.
    """

    def __init__(self, settings, history):
        self.settings = settings
        self.history = history
        self.hard_stop = False # hard break of code

        # indicates if we are connecting to server(s)
        if self.settings.client_server:
            # init Connection

            self.server_ports_suggestions = self.settings.server_ports_list
            self.server_ports_list = []
            self.server_ports_number_of_requests = {}
            self.number_of_server_machines = 0
            self.server_names = {} # port -> server name

            # Split machines to attention and final evaluation
            # total N - self.reserve_machines_for_attention will go to final evaluation
            self.reserve_machines_for_attention = self.settings.reserve_machines_for_attention

            self.handshake()

            if self.settings.precompute_attention_evaluation:
                self.prepare_attention_evaluation_server_lists()

            if self.settings.debug_just_handshake:
                self.hard_stop = True

        self.pool = ThreadPool()


        self.times_del = []

    def handshake(self):
        if self.settings.verbosity >= 2:
            print("Connection init, handshake")

        servers_dedicated_for_precompute = self.reserve_machines_for_attention
        failed_on_ports = []
        for port in self.server_ports_suggestions:
            try:
                HANDSHAKE_API_URL = "http://localhost:" + port + "/handshake"
                backup_name = port
                payload = {"client": "Hi, I am Bob.", "backup_name": backup_name}

                start = timer()
                r = requests.post(HANDSHAKE_API_URL, files=payload).json()
                end = timer()

                if r["success"]:
                    if self.settings.verbosity >= 1:
                        print("Connection with server on port",port,"established. Time:", (end - start), "Request data:", r)

                    server_name = r["server_name"]

                    if self.settings.precompute_attention_evaluation:
                        if servers_dedicated_for_precompute > 0:
                            server_name += '_Att'
                            servers_dedicated_for_precompute -= 1

                    self.server_names[port] = server_name

                    self.server_ports_list.append(port)
                    self.server_ports_number_of_requests[port] = 0
                else:
                    failed_on_ports.append(port)

            except Exception:
                failed_on_ports.append(port)

        if self.settings.verbosity >= 1:
            print("FAILED on ports:", " ".join(failed_on_ports))
            print("SUCCESS on ports:", " ".join(self.server_ports_list))
            print("machine names:", self.server_names)

        self.number_of_server_machines = len(self.server_ports_list)
        if (self.number_of_server_machines == 0):
            print("Connection to all servers failed! Backup solution = turning to local evaluation, no precomputing allowed.")
            self.settings.client_server = False
            self.settings.precompute_attention_evaluation = False
        if (self.number_of_server_machines < 2):
            print("Only one server connected! No precomputing allowed.")
            self.settings.precompute_attention_evaluation = False

        # limiter
        if self.settings.final_evaluation_limit_servers > 0:
            t = 0
            if self.settings.precompute_attention_evaluation:
                t = self.reserve_machines_for_attention
            for_final = len(self.server_ports_list) - t
            #print(for_final,"vs",self.settings.final_evaluation_limit_servers)

            if self.settings.final_evaluation_limit_servers < for_final:
                self.server_ports_list = self.server_ports_list[0:self.settings.final_evaluation_limit_servers+t]
                print("Limiting number of final-evaluation-servers to",len(self.server_ports_list),":",self.server_ports_list)
                self.number_of_server_machines = len(self.server_ports_list)


    def prepare_attention_evaluation_server_lists(self):
        N = self.number_of_server_machines

        if N > 1:
            # now we can split
            self.attention_machines_ports = []
            self.evaluation_machines_ports = []

            for i in range(self.reserve_machines_for_attention):
                self.attention_machines_ports.append(self.server_ports_list[i])

            for i in range(self.reserve_machines_for_attention, N):
                self.evaluation_machines_ports.append(self.server_ports_list[i])

            if self.settings.verbosity >= 1:
                print("attention_machines_ports", self.attention_machines_ports)
                print("evaluation_machines_ports", self.evaluation_machines_ports)

    def evaluate_crops_on_server(self, crops, ids_of_crops, type):
        # will be more advanced
        # like splitting to all available servers
        N = self.number_of_server_machines

        if N > 1:
            #print("ON MULTIPLE MACHINES", N, "(att:",len(self.attention_machines_ports),", eval:",len(self.evaluation_machines_ports),")")
            result, times_encode, times_eval, times_decode, times_transfer = self.split_across_list_of_servers(crops, ids_of_crops, type)
        else:
            port = self.server_ports_list[0]
            result,time_Encode, time_Evaluation, time_Decode, time_Transfer = self.direct_to_server(crops, ids_of_crops, port)
            times_encode = [time_Encode]
            times_eval = [time_Evaluation]
            times_decode = [time_Decode]
            times_transfer = [time_Transfer]
        return result, times_encode, times_eval, times_decode, times_transfer

    def split_across_list_of_servers(self, crops, ids_of_crops, type):

        # Scheduling task, idea/heuristic:
        #
        # hold two lists of servers reserved for two tasks:
        # - regular evaluation
        # - precomputing (attention) evaluation in advance
        # simple version, from N servers:
        # - 1 precomputing          // scale -> 2 ...
        # - N-1 final evaluations   // scale -> N-2 ...
        # if N=1, then use the direct method
        #
        # we want to prevent:
        #  a.) precomputing slowing down real evaluation
        #  b.) precomputing not being done fast enough

        ports_list = []
        if self.settings.precompute_attention_evaluation:
            if type == 'attention':
                ports_list = self.attention_machines_ports
            elif type == 'evaluation':
                ports_list = self.evaluation_machines_ports
        else:
            ports_list = self.server_ports_list

        shuffle(ports_list)

        N = len(ports_list)
        C = len(crops)

        if self.settings.verbosity >= 3:
            print("[Connection to multiple servers] We can split",C,"crops on",N," machines as ",(C/N)," per each (type",type,")")
            print("all ids:",ids_of_crops)

        results = [[]]*(max(ids_of_crops)+1)
        threads = []

        id_of_indices_0_to_C = range(0,C)
        id_splits = np.array_split(id_of_indices_0_to_C, N)
        #print("id_splits",id_splits)

        num_of_actual_threads = 0
        for ids in id_splits:
            if len(ids)>0:
                num_of_actual_threads+=1

        times_eval = [[]]*(num_of_actual_threads)
        times_transfer = [[]]*(num_of_actual_threads)
        times_encode = [[]]*(num_of_actual_threads)
        times_decode = [[]]*(num_of_actual_threads)

        #print("corresponds to crops", np.array_split(ids_of_crops, N))

        for i,ids in enumerate(id_splits):
            if len(ids)==0:
                continue

            sub_crops = [crops[id] for id in ids]
            sub_ids = [ids_of_crops[id] for id in ids]
            port = ports_list[i]

            if self.settings.verbosity >= 3:
                print(port, self.server_names[port], "> with len=",len(sub_crops), "of ids:", sub_ids)

            # start a new thread to call the API
            t = Thread(target=self.eval_subset_and_save_to_list, args=(sub_crops, sub_ids, port, results, i, times_eval, times_transfer, times_encode, times_decode))
            t.daemon = True
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if self.settings.verbosity >= 3:
            print("All threads finished, assuming we have all the ids from", ids_of_crops)

        results_tmp = []

        for i,r in enumerate(results):
            if len(r) > 0:
                results_tmp.append(r)

        results = results_tmp

        if self.settings.verbosity >= 3:
            print("results = ", results)

        return results, times_encode, times_eval, times_decode, times_transfer

    # thread function
    def eval_subset_and_save_to_list(self, crops, ids_of_crops, port, results, ith, times_eval, times_transfer, times_encode, times_decode):
        evaluation,time_Encode, time_Evaluation, time_Decode, time_Transfer = self.direct_to_server(crops, ids_of_crops, port)
        times_eval[ith] = time_Evaluation
        times_transfer[ith] = time_Transfer
        times_encode[ith] = time_Encode
        times_decode[ith] = time_Decode
        #print("times[ith]", ith, " = ", time)

        for uid,bbox in evaluation:
            results[uid] = [uid,bbox]
            #results[uid] = [uid]

        self.history.report_evaluation_per_specific_server(self.server_names[port], time_Encode, time_Evaluation, time_Decode, time_Transfer)


    def direct_to_server(self, crops, ids_of_crops, port):

        EVALUATE_API_URL = "http://localhost:" + port + "/evaluate_image_batch"

        number_of_images = len(crops)

        payload = {}


        t0 = timer()
        encoded_images = self.pool.map(lambda i: (
            cv2.imencode('.jpg', i)[1].tostring()
        ), crops)

        for i in range(number_of_images):
            #image = crops[i]
            #image_enc = cv2.imencode('.jpg', image)[1].tostring()
            image_enc = encoded_images[i]
            id = ids_of_crops[i]
            payload[str(id)] = image_enc

            """
            if self.settings.opencv_or_pil != 'PIL':
                # TODO: MAYBE INEFFICIENT, back to PIL for sending
                image = Image.fromarray(image)

            memory_file = BytesIO()
            image.save(memory_file, "JPEG")
            memory_file.seek(0)

            id = ids_of_crops[i]
            payload[str(id)] = memory_file
            """
        t1 = timer()
        time_Encode = t1-t0
        if self.settings.verbosity >= 2:
            print(number_of_images,"Image(s) encoding (with",self.settings.opencv_or_pil,") took = ", time_Encode, "(during the eval)")

        if number_of_images == 0:
            print("Careful, 0 images, don't send.")
            return [],0

        start = timer()
        # submit the request
        try:
            r = requests.post(EVALUATE_API_URL, files=payload).json()
        except Exception as e:
            print("CONNECTION TO SERVER ",EVALUATE_API_URL," FAILED - return to backup local evaluation?")
            print("Exception:", e)

        end = timer()
        time_EvaluationAndTransfer = end - start
        if self.settings.verbosity >= 2:
            print("Server on port",port," evaluated", len(crops), "crops. Time:", time_EvaluationAndTransfer, "Request data:", r)

        uids = r["uids"]
        bboxes = r["bboxes"]

        time_Evaluation = float(r["time_pure_eval"])
        time_Decode = float(r["time_pure_decode"])
        time_Transfer = time_EvaluationAndTransfer - time_Evaluation - time_Decode

        #print("uids", uids)
        #print("bboxes len", len(bboxes))
        # currently UIDS are ordered, this will not be true of multiple servers though

        evaluation = []
        for i,bbox in enumerate(bboxes):
            evaluation.append([int(uids[i]), bbox])

        # We want evaluation in format:
        #     array of crops in order by coordinates_id
        #     each holds id and array of dictionaries for each bbox {} keys label, confidence, topleft, bottomright
        #print("evaluation", evaluation)

        return evaluation, time_Encode, time_Evaluation, time_Decode, time_Transfer

