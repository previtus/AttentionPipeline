# From folder of full res. images generate smaller images depending on their kmeans
# also save the kmeans

import os
from PIL import Image

from helpers import make_dir_if_doesnt_exist

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from timeit import default_timer as timer

def offset_into_interval(values, interval):
    if values[1]>interval[1]:
        off = values[1]-interval[1]
        values[0] -= off
        values[1] -= off
    if values[0]<interval[0]:
        off = interval[0]-values[0]
        values[0] += off
        values[1] += off
    return values

# paths:
images_folder = "/home/ekmek/saliency_tools/_sample_inputs/images/"
saliency_folder = "/home/ekmek/saliency_tools/_sample_inputs/heatmaps/"
save_indices_folder = "/home/ekmek/saliency_tools/_sample_inputs/indices/"
make_dir_if_doesnt_exist(save_indices_folder)
save_plots2_folder = "/home/ekmek/saliency_tools/_sample_inputs/plots2/"
make_dir_if_doesnt_exist(save_plots2_folder)

save_crops_folder = "/home/ekmek/saliency_tools/_sample_inputs/crops/"
make_dir_if_doesnt_exist(save_crops_folder)

image_files = os.listdir(images_folder)
saliency_files = os.listdir(saliency_folder)

dictionary_id_to_centers = {}

start = timer()
#for i in range(1,2):
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

    tmp = Image.new(sal.mode, sal.size)
    tmp.putdata(pixels)
    px2d = np.array(tmp)
    nonzero_px = np.transpose(np.nonzero(px2d))

    k = 3
    kmeans = KMeans(n_clusters=k).fit(nonzero_px)

    centers = kmeans.cluster_centers_
    centers = centers.astype(int)

    fig, ax = plt.subplots()

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 1], centroids[:, 0],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)

    id = image_files[i][:-4]
    #print id, centers

    plt.imshow(img)
    plt.xlim(-100, 740)
    plt.ylim(-100, 740)
    plt.gca().invert_yaxis()

    desired_size = (224,224)
    k_idx = 0
    for center in centers:
        print center
        xc = center[0]
        yc = center[1]
        xleft = xc - desired_size[0]/2
        xright = xc + desired_size[0]/2
        ybottom = yc - desired_size[0]/2
        ytop = yc + desired_size[0]/2

        [xleft, xright] = offset_into_interval([xleft, xright], [0,608])
        [ybottom, ytop] = offset_into_interval([ybottom, ytop], [0,640])

        start_x = ybottom
        start_y = xleft
        width = 224
        height = 224
        area = (start_x, start_y, start_x + width, start_y + height)
        cropped_img = img.crop(box=area)
        cropped_img.load()
        print cropped_img, cropped_img.size
        #cropped_img.show()
        cropped_img.save(save_crops_folder+id+'_'+str(k_idx)+".jpg")

        ax.add_patch(
            patches.Rectangle(
                (ybottom, xleft),
                xright-xleft,
                ytop-ybottom,fill=False
            )
        )

        k_idx += 1
    #plt.show()

    plt.title(str(k)+'-means of image with threshold of '+str(int(perc*100))+'%')
    plt.savefig(save_plots2_folder+image_files[i],bbox_inches='tight', pad_inches=0, dpi=140)
    plt.close()

    dictionary_id_to_centers[id] = centers


end = timer()
t = (end - start)
print "total time:", t, ", per img:", (t/(i+1))


