# Save mixes into new folder = alpha blended image and saliency over it

import os
from PIL import Image

from helpers import make_dir_if_doesnt_exist

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np



# paths:
images_folder = "/home/ekmek/saliency_tools/_sample_inputs/images/"
saliency_folder = "/home/ekmek/saliency_tools/_sample_inputs/heatmaps/"
save_mixes_folder = "/home/ekmek/saliency_tools/_sample_inputs/mixes/"
make_dir_if_doesnt_exist(save_mixes_folder)

image_files = os.listdir(images_folder)
saliency_files = os.listdir(saliency_folder)

#for i in range(0,1):
for i in range(0,len(image_files)):
    img_path = images_folder+image_files[i]
    sal_path = saliency_folder+saliency_files[i]

    img = Image.open(img_path)
    sal = Image.open(sal_path).convert("RGB").resize(img.size, Image.ANTIALIAS)

    print img.mode, img.size, sal.mode, sal.size

    #img.show()
    #sal.show()

    mix = Image.blend(img, sal, alpha=0.9)

    mix.save(save_mixes_folder+image_files[i])
    #mix.show()

    #print img, sal