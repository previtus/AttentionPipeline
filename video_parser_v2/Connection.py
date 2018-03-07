import requests
from timeit import default_timer as timer
from evaluation_code.encoding import base64_encode_image

class Connection(object):
    """
    Holds connection to server(s), handles sending and receiving to and from the right one.
    """

    def __init__(self, settings):
        self.settings = settings

        # indicates if we are connecting to server(s)
        if self.settings.client_server:
            # init Connection

            self.server_ports_suggestions = self.settings.server_ports_list
            self.server_ports_list = []
            self.number_of_server_machines = 0

            self.handshake()

    def handshake(self):
        if self.settings.verbosity >= 2:
            print("Connection init, handshake")

        for port in self.server_ports_suggestions:
            try:
                HANDSHAKE_API_URL = "http://localhost:" + port + "/handshake"
                payload = {"client": "Hi, I am Bob."}

                start = timer()
                r = requests.post(HANDSHAKE_API_URL, files=payload).json()
                end = timer()

                if r["success"]:
                    if self.settings.verbosity >= 1:
                        print("Connection with server on port",port,"established. Time:", (end - start), "Request data:", r)
                    self.server_ports_list.append(port)
                else:
                    if self.settings.verbosity >= 1:
                        print("Connection with server on port", port, "FAILED.")

            except Exception:
                if self.settings.verbosity >= 1:
                    print("Connection with server on port", port, "FAILED.")

        self.number_of_server_machines = len(self.server_ports_list)
        if (self.number_of_server_machines == 0):
            print("Connection to all servers failed! Backup solution = turning to local evaluation.")
            self.settings.client_server = False

    def evaluate_crops_on_server(self, crops, ids_of_crops):
        # will be more advanced
        # like splitting to all available servers
        return self.direct_to_first_server(crops, ids_of_crops)

    def direct_to_first_server(self, crops, ids_of_crops):
        port = self.server_ports_list[0]

        EVALUATE_API_URL = "http://localhost:" + port + "/evaluate_image_batch"

        number_of_images = len(crops)

        start = timer()
        payload = {}
        for i in range(number_of_images):
            image = crops[i]
            image = base64_encode_image(image)
            id = ids_of_crops[i]
            payload[str(id)] = image

        # submit the request
        r = requests.post(EVALUATE_API_URL, files=payload).json()

        end = timer()
        t = end - start
        if self.settings.verbosity >= 2:
            print("Server evaluated", len(crops), "crops. Time:", t, "Request data:", r)

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

        return evaluation

