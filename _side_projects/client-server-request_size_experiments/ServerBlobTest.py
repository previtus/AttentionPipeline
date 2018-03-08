import evaluation_code.darkflow_handler as darkflow_handler
from keras.preprocessing.image import img_to_array
from evaluation_code.encoding import base64_decode_image

from threading import Thread
import time

from PIL import Image
import flask
import io
import os
from timeit import default_timer as timer
from multiprocessing.pool import ThreadPool

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

app = flask.Flask(__name__)
darkflow_model = None
pool = ThreadPool()

# del
from timeit import default_timer as timer
import numpy as np
import base64
import sys


times_del = []

def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a


class Server(object):
    """
    Server
    """

    def __init__(self):
        print("Server ... starting server and loading model ... please wait until its started ...")

        frequency_sec = 25.0
        t = Thread(target=self.mem_monitor_deamon, args=([frequency_sec]))
        t.daemon = True
        t.start()

        app.run()
        # On server:
        #app.run(host='0.0.0.0', port=8123)

    def mem_monitor_deamon(self, frequency_sec):
        import subprocess
        while (True):
            out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                                   stdout=subprocess.PIPE).communicate()[0].split(b'\n')
            vsz_index = out[0].split().index(b'RSS')
            mem = float(out[1].split()[vsz_index]) / 1024

            print("Memory:", mem)
            time.sleep(frequency_sec)  # check every frequency_sec sec

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

@app.route("/test_request_size", methods=["POST"])
def test_request_size():
    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        lens = []
        blobs = []

        t1 = timer()

        for key in flask.request.files:
            blob = flask.request.files[key].read()
            l = len(blob)
            print("blobl length/size",l)
            lens.append(l)
            blobs.append(blob)

        t2 = timer()
        times_del.append((t2-t1))
        print("reading time avg ", np.mean(times_del))

        print("trying to load it...")
        blob = blobs[0]
        print("blob : ", type(blob), blob)
        image = Image.open(io.BytesIO(blob))
        print("image : ", type(image), image)
        image = img_to_array(image)
        print("image : ", type(image), image.shape)

        data["time"] = (t2-t1)
        data["blob_lengths"] = lens

        # indicate that the request was a success
        data["success"] = True

    return flask.jsonify(data)



if __name__ == "__main__":
    server = Server()