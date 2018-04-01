import evaluation_code.darkflow_handler as darkflow_handler
from evaluation_code.encoding import base64_decode_image

from threading import Thread
import time

from PIL import Image
import flask
import io
import os
from timeit import default_timer as timer
from multiprocessing.pool import ThreadPool
import numpy as np
import socket

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

app = flask.Flask(__name__)
darkflow_model = None
pool = ThreadPool()

# del
from timeit import default_timer as timer
import numpy

times_del = []

class Server(object):
    """
    Server
    """

    def __init__(self):
        print("Server ... starting server and loading model ... please wait until its started ...")
        self.warm_up = 0

        self.load_model_darkflow()

        frequency_sec = 10.0
        t = Thread(target=self.mem_monitor_deamon, args=([frequency_sec]))
        t.daemon = True
        t.start()

        # hack to distinguish server
        # this might not work on non gpu machines
        # but we are using only those
        hostname = socket.gethostname()  # gpu048.etcetcetc.edu
        if hostname[0:3] == "gpu":
            app.run(host='0.0.0.0', port=8123)
        else:
            app.run(port=5001)

    def mem_monitor_deamon(self, frequency_sec):
        import subprocess
        while (True):
            out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                                   stdout=subprocess.PIPE).communicate()[0].split(b'\n')
            vsz_index = out[0].split().index(b'RSS')
            mem = float(out[1].split()[vsz_index]) / 1024

            print("Memory:", mem)
            time.sleep(frequency_sec)  # check every frequency_sec sec

    def load_model_darkflow(self):
        global darkflow_model
        darkflow_model = darkflow_handler.load_model(self.warm_up)
        print('Model loaded.')



@app.route("/handshake", methods=["POST"])
def handshake():
    # Handshake

    data = {"success": False}
    start = timer()

    if flask.request.method == "POST":
        if flask.request.files.get("client"):
            client_message = flask.request.files["client"].read()
            print("Handshake, received: ",client_message)

            backup_name = flask.request.files["backup_name"].read()
            # try to figure out what kind of server we are, what is our name, where do we live, what are we like,
            # which gpu we occupy
            # and return it at an identifier to the client ~


            try:
                hostname = socket.gethostname() # gpu048.etcetcetc.edu
                machine_name = hostname.split(".")[0]
                buses = get_gpus_buses()
                print("Bus information =",buses)
                if len(buses) > 0:
                    buses = ":"+buses
                data["server_name"] = machine_name + buses
            except Exception as e:
                data["server_name"] = backup_name


            end = timer()
            data["internal_time"] = end - start
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route("/evaluate_image_batch", methods=["POST"])
def evaluate_image_batch():
    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        images = []
        uids = []

        t_start_eval = timer()
        t1 = timer()

        for key in flask.request.files:
            image = flask.request.files[key].read()
            image = Image.open(io.BytesIO(image))
            image = my_img_to_array(image) # maybe

            images.append(image)
            uids.append(key)

        t2 = timer()
        times_del.append((t2-t1))
        print("avg reading ", numpy.mean(times_del))

        print("Received",len(images),"images.", uids, [i.shape for i in images])

        results_bboxes = darkflow_handler.run_on_images(image_objects=images, model=darkflow_model)
        t_end_eval = timer()

        data["bboxes"] = results_bboxes
        data["uids"] = uids
        data["time_pure_eval"] = t_end_eval-t_start_eval

        # indicate that the request was a success
        data["success"] = True

    return flask.jsonify(data)

def my_img_to_array(img):
    # remove Keras dep
    x = np.asarray(img, dtype='float32')
    return x

from tensorflow.python.client import device_lib
def get_gpus_buses():
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [x for x in local_device_protos if x.device_type == 'GPU']
    buses = ""
    for device in gpu_devices:
        desc = device.physical_device_desc # device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:81:00.0
        bus = desc.split(",")[-1].split(" ")[-1][5:] # split to get to the bus information
        bus = bus[0:2] # idk if this covers every aspect of gpu bus
        if len(buses)>0:
            buses += ";"
        buses += str(bus)
    return buses


if __name__ == "__main__":
    server = Server()