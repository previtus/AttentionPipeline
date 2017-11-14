# Save coordinates of the kmeans

import os
from PIL import Image

from helpers import make_dir_if_doesnt_exist

import numpy as np
from sklearn.cluster import KMeans

from timeit import default_timer as timer

# paths:
images_folder = "/home/ekmek/saliency_tools/_sample_inputs/images/"
saliency_folder = "/home/ekmek/saliency_tools/_sample_inputs/heatmaps/"
save_indices_folder = "/home/ekmek/saliency_tools/_sample_inputs/indices/"
make_dir_if_doesnt_exist(save_indices_folder)

image_files = os.listdir(images_folder)
saliency_files = os.listdir(saliency_folder)

dictionary_id_to_centers = {}

start = timer()
#for i in range(0,1):
for i in range(0,len(image_files)):
    img_path = images_folder+image_files[i]
    sal_path = saliency_folder+saliency_files[i]

    img = Image.open(img_path)
    sal = Image.open(sal_path).convert("L")
    pixels = list(sal.getdata())
    pixels = np.array(pixels)
    perc = 0.5
    threshold = int((pixels.max() - pixels.min())*perc)

    pixels[pixels < threshold] = 0
    #pixels[pixels >= threshold] = 256

    tmp = Image.new(sal.mode, sal.size)
    tmp.putdata(pixels)
    #tmp.show()
    px2d = np.array(tmp)
    nonzero_px = np.transpose(np.nonzero(px2d))

    k = 4
    kmeans = KMeans(n_clusters=k).fit(nonzero_px)

    centers = kmeans.cluster_centers_
    centers = centers.astype(int)

    id = image_files[i][:-4]
    print id

    dictionary_id_to_centers[id] = centers

np.save(save_indices_folder+'indices.npy', dictionary_id_to_centers)

end = timer()
t = (end - start)
print "total time:", t, ", per img:", (t/(i+1))


