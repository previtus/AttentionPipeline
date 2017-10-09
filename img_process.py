# process heatmaps

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.pyplot import imshow

from visualization import visualize_kmeans

import numpy as np
from sklearn.cluster import KMeans

from timeit import default_timer as timer

# paths:
images_folder = "/home/ekmek/saliency_tools/_sample_inputs/images/"
saliency_folder = "/home/ekmek/saliency_tools/_sample_inputs/heatmaps/"
save_mixes_folder = "/home/ekmek/saliency_tools/_sample_inputs/mixes/"
save_plots_folder = "/home/ekmek/saliency_tools/_sample_inputs/plots/"

image_files = os.listdir(images_folder)
saliency_files = os.listdir(saliency_folder)

start = timer()
#for i in range(0,3):
for i in range(0,len(image_files)):
    img_path = images_folder+image_files[i]
    sal_path = saliency_folder+saliency_files[i]

    img = Image.open(img_path)
    sal = Image.open(sal_path).convert("L")
    pixels = list(sal.getdata())
    pixels = np.array(pixels)
    perc = 0.5
    threshold = int((pixels.max() - pixels.min())*perc)

    print img_path, pixels.min(), pixels.max(), threshold

    pixels[pixels < threshold] = 0
    #pixels[pixels >= threshold] = 256

    tmp = Image.new(sal.mode, sal.size)
    tmp.putdata(pixels)
    #tmp.show()
    px2d = np.array(tmp)
    nonzero_px = np.transpose(np.nonzero(px2d))
    print len(nonzero_px), nonzero_px[0:10]

    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    print X.shape
    print nonzero_px.shape

    k = 4
    kmeans = KMeans(n_clusters=k).fit(nonzero_px)
    print kmeans
    print kmeans.cluster_centers_
    print kmeans.labels_

    plt = visualize_kmeans(kmeans, nonzero_px, background_img=img, show=False)
    plt.title(str(k)+'-means of image with threshold of '+str(int(perc*100))+'%')
    plt.savefig(save_plots_folder+image_files[i],bbox_inches='tight', pad_inches=0, dpi=140)
    plt.close()

end = timer()
t = (end - start)
print "total time:", t, ", per img:", (t/(i+1))


