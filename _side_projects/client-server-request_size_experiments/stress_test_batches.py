# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time
from timeit import default_timer as timer
from PIL import Image
from keras.preprocessing.image import img_to_array

import numpy as np
import base64
import sys

def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")



PORT = "5000"
YOLO_KERAS_REST_API_URL = "http://localhost:"+PORT+"/test_request_size"

NUM_REQUESTS = 50
C = 10

from io import BytesIO

def various_image_loads(n, image_paths):
	image_index = n % len(images)

	# load the input image and construct the payload for the request
	start = timer()

	payload = {}
	load = 3
	if load==1:
		for i in range(C):
			image = Image.open(image_paths[image_index+i])
			image = img_to_array(image)
			encoded_img = base64_encode_image(image)
			payload[str(n*C + i)] = encoded_img

	elif load==2:
		for i in range(C):
			image = open(image_paths[image_index+i], "rb").read()
			payload[str(n * C + i)] = image

	elif load==3:
		for i in range(C):
			#PIL img
			image = Image.open(image_paths[image_index+i])

			memory_file = BytesIO()
			image.save(memory_file, "JPEG")
			memory_file.seek(0)

			payload[str(n * C + i)] = memory_file


	# submit the request
	r = requests.post(YOLO_KERAS_REST_API_URL, files=payload).json()

	end = timer()
	t = end - start

	# ensure the request was sucessful
	if r["success"]:
		print("[INFO] thread {} OK".format(n), t, " one ", t/float(C), "{...",image_paths[image_index][-20:],"}")
		times.put([n,t])

	# otherwise, the request failed
	else:
		print("[INFO] thread {} FAILED".format(n), t)

import queue
times = queue.Queue()
threads = []

import os, fnmatch, random
path = "/home/ekmek/python_codes/1K_square_imgs/"
files = sorted(os.listdir(path))
frame_files = fnmatch.filter(files, '*.jpg')
images = [path + i for i in frame_files]
random.shuffle(images)
#print(images)

for i in range(0, NUM_REQUESTS):
	# start a new thread to call the API
	image_index = i%len(images)

	various_image_loads(i, images)

	#t = Thread(target=call_predict_endpoint, args=(i,images[image_index]))
	#t.daemon = True
	#t.start()
	#threads.append(t)
	#time.sleep(SLEEP_COUNT)

for t in threads:
	t.join()
