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

        gpu_num = 2.0 # or 1.0
        # with 6 jobs, 1 gpu each worked with 1.0 and CUDA_VISIBLE_DEVICES=0
        # with 3 jobs, 2gpus each worked with 2.0 and CUDA_VISIBLE_DEVICES=1

        """
        whoops:
        self.define('gpu', 0.0, 'how much gpu (from 0.0 to 1.0)')
        self.define('gpuName', '/gpu:0', 'GPU device name')
        """

        self.load_model_darkflow(gpu_num)

        frequency_sec = 10.0
        t = Thread(target=self.mem_monitor_deamon, args=([frequency_sec]))
        t.daemon = True
        t.start()

        #app.run()
        # On server:
        app.run(host='0.0.0.0', port=8666)

    def mem_monitor_deamon(self, frequency_sec):
        import subprocess
        while (True):
            out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                                   stdout=subprocess.PIPE).communicate()[0].split(b'\n')
            vsz_index = out[0].split().index(b'RSS')
            mem = float(out[1].split()[vsz_index]) / 1024

            print("Memory:", mem)
            time.sleep(frequency_sec)  # check every frequency_sec sec

    def load_model_darkflow(self, gpu_num):
        global darkflow_model
        darkflow_model = darkflow_handler.load_model(gpu_num)
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

        data["bboxes"] = results_bboxes
        data["uids"] = uids

        # indicate that the request was a success
        data["success"] = True

    return flask.jsonify(data)

def my_img_to_array(img):
    # remove Keras dep
    x = np.asarray(img, dtype='float32')
    return x

if __name__ == "__main__":
    server = Server()