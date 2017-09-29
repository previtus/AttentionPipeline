# show figures of images

import os
from PIL import Image

#from matplotlib.pyplot import imshow
#import numpy as np

# paths:
images_folder = "/home/ekmek/saliency-salgan-2017/images/"
saliency_folder = "/home/ekmek/saliency-salgan-2017/saliency/"

image_files = os.listdir(images_folder)
saliency_files = os.listdir(saliency_folder)

for i in [0]: #range(0,len(image_files)):
    img_path = images_folder+image_files[i]
    sal_path = saliency_folder+saliency_files[i]

    img = Image.open(img_path)
    sal = Image.open(sal_path).convert("RGB")

    print img.mode, img.size, sal.mode, sal.size

    #img.show()
    #sal.show()

    mix = Image.blend(img, sal, alpha=0.9)

    mix.show()
    #imshow(np.asarray(mix))

    print img, sal