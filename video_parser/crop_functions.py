import os
import numpy, random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_crops_parameters(w, crop=288, over=0.5, scale=1.0):
    crop = scale * crop

    block = crop * (1.0 - over)
    pocet = (w - (crop - block)) / block
    nastejne = (w - (crop - block)) / int(pocet)

    offset = w - (int((int(pocet) - 1) * nastejne) + crop)
    balance = offset / 2.0

    params = []
    for i in range(0, int(pocet)):
        w_from = int(i * nastejne + balance)
        w_to = int(w_from + crop)
        params.append((w_from, w_to))

    #print w - w_to
    return params

def crop_from_one_frame(frame_path, out_folder, crop, over, scale, show):
    # crop*scale is the size inside input image
    # crop is the size of output image
    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]

    if not os.path.exists(out_folder+frame_name+"/"):
        os.makedirs(out_folder+frame_name+"/")

    print frame_name, out_folder+frame_name+"/"

    img = Image.open(frame_path)
    width, height = img.size

    if show:
        fig, ax = plt.subplots()

        plt.imshow(img)
        plt.xlim(-1 * (width / 10.0), width + 1 * (width / 10.0))
        plt.ylim(-1 * (height / 10.0), height + 1 * (height / 10.0))
        plt.gca().invert_yaxis()

    w_crops = get_crops_parameters(width, crop, over, scale)
    h_crops = get_crops_parameters(height, crop, over, scale)
    N = len(w_crops) * len(h_crops)

    print "Number of crops:", N

    i = 0
    for w_crop in w_crops:
        for h_crop in h_crops:
            if show:
                jitter = random.uniform(0, 1) * 15

                ax.add_patch(
                    patches.Rectangle(
                        (w_crop[0] + jitter, h_crop[0] + jitter),
                        scale * crop,
                        scale * crop, fill=False, linewidth=2.0, color=numpy.random.rand(3, 1)  # color=cmap(i)
                    )
                )

            area = (w_crop[0], h_crop[0], w_crop[0] + scale * crop, h_crop[0] + scale * crop)
            cropped_img = img.crop(box=area)
            cropped_img = cropped_img.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_img.load()
            cropped_img.save(out_folder+frame_name+"/"+ str(i) + ".jpg")
            i += 1

    if show:
        plt.show()