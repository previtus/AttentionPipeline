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
import numpy

times_del = []

class Server(object):
    """
    Server
    """

    def __init__(self):
        print("Server ... starting server and loading model ... please wait until its started ...")

        gpu_num = 1.0
        self.load_model_darkflow(gpu_num)

        frequency_sec = 10.0
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
            image = img_to_array(image) # maybe

            #image = image.decode("utf-8")

            #image = base64_decode_image(image,dtype="float32",shape=(1, 608, 608,3))
            #if image.shape[0] != 1:
            #    print("CAREFUL, i made wrong assumption about image.shape", image.shape, "its not 1")
            #image = image[0]

            images.append(image)
            uids.append(key)

        t2 = timer()
        times_del.append((t2-t1))
        print("avg reading ", numpy.mean(times_del))

        #decoded_images = pool.map(lambda img: (
        #    base64_decode_image(img.decode("utf-8"), dtype="float32", shape=(1, 608, 608, 3))
        #), images)
        #images = [img[0] for img in decoded_images]

        print("Received",len(images),"images.", uids, [i.shape for i in images])

        results_bboxes = darkflow_handler.run_on_images(image_objects=images, model=darkflow_model)

        data["bboxes"] = results_bboxes
        data["uids"] = uids

        # indicate that the request was a success
        data["success"] = True

    return flask.jsonify(data)


if __name__ == "__main__":
    server = Server()