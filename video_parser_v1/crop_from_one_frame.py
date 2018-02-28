import os
import numpy, random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from crop_functions import get_crops_parameters

img_path = "/home/ekmek/intership_project/video_parser_v1/PL_Pizza/0000.jpg"
save_crops_folder = "/home/ekmek/intership_project/video_parser_v1/crops_test/"
if not os.path.exists(save_crops_folder):
    os.makedirs(save_crops_folder)

img = Image.open(img_path)
width, height = img.size

bool_generate_plot = True

if bool_generate_plot:
    fig, ax = plt.subplots()

    plt.imshow(img)
    plt.xlim(-1*(width/10.0), width+1*(width/10.0))
    plt.ylim(-1*(height/10.0), height+1*(height/10.0))
    plt.gca().invert_yaxis()


# crop*scale is the size inside input image
# crop is the size of output image
crop = 544
over = 0.6
scale = 1.0
w_crops = get_crops_parameters(width, crop, over, scale)
h_crops = get_crops_parameters(height, crop, over, scale)

print ("Number of crops:", len(w_crops) * len(h_crops))

N = len(w_crops) * len(h_crops)
cmap = plt.cm.get_cmap("hsv", N+1)

i = 0
for w_crop in w_crops:
    for h_crop in h_crops:
        jitter = random.uniform(0, 1) * 15

        ax.add_patch(
            patches.Rectangle(
                (w_crop[0] + jitter, h_crop[0] + jitter),
                scale*crop,
                scale*crop, fill=False, linewidth=2.0, color=numpy.random.rand(3,1) #color=cmap(i)
            )
        )

        area = (w_crop[0], h_crop[0], w_crop[0] + scale*crop, h_crop[0] + scale*crop)
        cropped_img = img.crop(box=area)
        cropped_img = cropped_img.resize((crop,crop),resample=Image.ANTIALIAS)
        cropped_img.load()
        cropped_img.save(save_crops_folder+'_'+str(i)+".jpg")
        i += 1


"""
width = 3840
height = 2160

crop_sizes_possible = [288,352,416,480,544]

"""

plt.show()