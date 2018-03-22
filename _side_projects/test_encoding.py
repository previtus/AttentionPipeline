from PIL import Image
from keras.preprocessing.image import img_to_array
import io
import os

import requests
from timeit import default_timer as timer
from evaluation_code.encoding import base64_encode_image, base64_decode_image

path = "/home/ekmek/intership_project/video_parser_v2/__Renders/_Test-6mar/0001.jpg"
img = Image.open(path)
cropped_img = img.crop([0,0,608,608])
if cropped_img.size[0] != 608 or cropped_img.size[1] != 608:
    print("Careful, needed to resize the crop in Evaluation->ImageProcessing. It was", cropped_img.size)
    cropped_img = cropped_img.resize((608, 608), resample=Image.ANTIALIAS)
cropped_img.load()

cropped_img = img_to_array(cropped_img)

port = "5000"
EVALUATE_API_URL = "http://localhost:" + port + "/evaluate_image_batch"


payload = {}
image = cropped_img
print("image", type(image))
#image = image.copy(order="C")  # ?
print("image", type(image))
image = base64_encode_image(image)  # ?
#image = image.tobytes('C')
print("image", type(image))


payload[str(1)] = image
payload[str(2)] = image
r = requests.post(EVALUATE_API_URL, files=payload).json()
print("--------------")

IMAGE_WIDTH = 608
IMAGE_HEIGHT = 608
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

#image = image.decode("utf-8")
#print("image", type(image))

image = base64_decode_image(image,IMAGE_DTYPE,(1, IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_CHANS))

print("image", type(image))
print(image.shape)
#image = Image.open(io.BytesIO(image))
# image = img_to_array(image)
