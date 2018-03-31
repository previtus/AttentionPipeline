import requests
from multiprocessing.pool import ThreadPool
from io import BytesIO
import math
from threading import Thread
import numpy as np

from PIL import Image
import cv2
from timeit import default_timer as timer

class Connection(object):
    """
    Holds connection to server(s), handles sending and receiving to and from the right one.
    """

    def __init__(self, settings):
        self.settings = settings
        self.hard_stop = False # hard break of code

        # indicates if we are connecting to server(s)
        if self.settings.client_server:
            # init Connection

            self.server_ports_suggestions = self.settings.server_ports_list
            self.server_ports_list = []
            self.server_ports_number_of_requests = {}
            self.number_of_server_machines = 0

            self.reserve_machines_for_attention = 1

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
                    self.server_ports_list.append(port)
                    self.server_ports_number_of_requests[port] = 0
                else:
                    failed_on_ports.append(port)

            except Exception:
                failed_on_ports.append(port)

        if self.settings.verbosity >= 1:
            print("FAILED on ports:", " ".join(failed_on_ports))
            print("SUCCESS on ports:", " ".join(self.server_ports_list))

        self.number_of_server_machines = len(self.server_ports_list)
        if (self.number_of_server_machines == 0):
            print("Connection to all servers failed! Backup solution = turning to local evaluation, no precomputing allowed.")
            self.settings.client_server = False
            self.settings.precompute_attention_evaluation = False
        if (self.number_of_server_machines < 2):
            print("Only one server connected! No precomputing allowed.")
            self.settings.precompute_attention_evaluation = False

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
            result,times = self.split_across_list_of_servers(crops, ids_of_crops, type)
        else:
            port = self.server_ports_list[0]
            result,time = self.direct_to_server(crops, ids_of_crops, port)
            times = [time]
        return result, times

        """
        port = self.server_ports_list[0]
        result2 = self.direct_to_server(crops, ids_of_crops, port)

        print("===SHOULD BE THE SAME")
        print("result1",len(result1), result1)
        print("result2",len(result2), result2)

        return result2
        """

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

        N = len(ports_list)
        C = len(crops)

        if self.settings.verbosity >= 3:
            print("[Connection to multiple servers] We can split",C,"crops on",N," machines as ",(C/N)," per each (type",type,")")
            print("all ids:",ids_of_crops)

        results = [[]]*(max(ids_of_crops)+1)
        times = [[]]*(N)
        threads = []

        id_splits = np.array_split(ids_of_crops, N)
        print("id_splits",id_splits)
        for i,ids in enumerate(id_splits):
            if len(ids)==0:
                continue

            sub_crops = [crops[id] for id in ids]
            sub_ids = [ids_of_crops[id] for id in ids]
            port = ports_list[i]

            if self.settings.verbosity >= 3:
                print(port, "> with len=",len(sub_crops), "of ids:", sub_ids)

            # start a new thread to call the API
            t = Thread(target=self.eval_subset_and_save_to_list, args=(sub_crops, sub_ids, port, results, i, times))
            t.daemon = True
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        if self.settings.verbosity >= 3:
            print("All threads finished, assuming we have all the ids from", ids_of_crops)

        results_tmp = []
        #print("times", times)

        for i,r in enumerate(results):
            if len(r) > 0:
                results_tmp.append(r)

        results = results_tmp

        if self.settings.verbosity >= 3:
            print("results = ", results)

        return results, times

    def eval_subset_and_save_to_list(self, crops, ids_of_crops, port, results, ith, times):
        evaluation,time = self.direct_to_server(crops, ids_of_crops, port)
        times[ith] = time
        #print("times[ith]", ith, " = ", time)

        for uid,bbox in evaluation:
            results[uid] = [uid,bbox]
            #results[uid] = [uid]

    def direct_to_server(self, crops, ids_of_crops, port):
        start = timer()

        EVALUATE_API_URL = "http://localhost:" + port + "/evaluate_image_batch"

        number_of_images = len(crops)

        payload = {}

        for i in range(number_of_images):
            image = crops[i]

            if self.settings.opencv_or_pil != 'PIL':
                # TODO: MAYBE INEFFICIENT, back to PIL for sending
                image = Image.fromarray(image)

            memory_file = BytesIO()
            image.save(memory_file, "JPEG")
            memory_file.seek(0)

            id = ids_of_crops[i]
            payload[str(id)] = memory_file
        t0 = timer()
        print("Image encoding (with",self.settings.opencv_or_pil,") took = ", t0-start, "(during the eval)")

        if number_of_images == 0:
            print("Careful, 0 images, don't send.")
            return [],0

        # submit the request
        try:
            r = requests.post(EVALUATE_API_URL, files=payload).json()
        except Exception as e:
            print("CONNECTION TO SERVER ",EVALUATE_API_URL," FAILED - return to backup local evaluation?")
            print("Exception:", e)

        end = timer()
        time = end - start
        if self.settings.verbosity >= 2:
            print("Server on port",port," evaluated", len(crops), "crops. Time:", time, "Request data:", r)

        print("request", r)

        uids = r["uids"]
        bboxes = r["bboxes"]

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

        return evaluation, time

