import os
import numpy as np
import random
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_crops_parameters(w, crop=288, over=0.5, scale=1.0):
    crop = scale * crop

    block = crop * (1.0 - over)
    pocet = (w - (crop - block)) / block
    pocet = max([pocet,1.0])
    nastejne = (w - (crop - block)) / int(pocet)

    offset = w - (int((int(pocet) - 1) * nastejne) + crop)
    balance = offset / 2.0

    params = []
    for i in range(0, int(pocet)):
        w_from = int(i * nastejne + balance)
        w_to = int(w_from + crop)
        params.append((w_from, w_to))

    #print (w - w_to)
    return params

def crop_from_one_img(img, crop, over, scale, show=False, save_crops=True, folder_name='', frame_name=''):

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

    crops = []
    i = 0
    for w_crop in w_crops:
        for h_crop in h_crops:
            if show:
                jitter = random.uniform(0, 1) * 15

                ax.add_patch(
                    patches.Rectangle(
                        (w_crop[0] + jitter, h_crop[0] + jitter),
                        scale * crop,
                        scale * crop, fill=False, linewidth=2.0, color=np.random.rand(3, 1)  # color=cmap(i)
                    )
                )

            area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + scale * crop), int(h_crop[0] + scale * crop))
            cropped_img = img.crop(box=area)
            cropped_img = cropped_img.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_img.load()

            if save_crops:
                file_name = frame_name + str(i).zfill(4) + ".jpg"
                cropped_img.save(folder_name + file_name)
            i += 1

            crops.append((file_name, area))
    if show:
        plt.show()

    return crops

def crop_from_one_frame(frame_path, out_folder, crop, over, scale, show, save_crops=True, save_visualization=True, viz_path=''):
    # crop*scale is the size inside input image
    # crop is the size of output image

    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]

    if not os.path.exists(out_folder+frame_name+"/"):
        os.makedirs(out_folder+frame_name+"/")

    img = Image.open(frame_path)

    """
    hack_resize_to_one_row = True
    if hack_resize_to_one_row:
        print(img.size)
        h = img.size[1]
        w = img.size[0]
        nh = crop
        nw = nh * w / h
        img = img.resize((int(nw), int(nh)), Image.ANTIALIAS)
        print(img.size)
    """
    width, height = img.size

    if show or save_visualization:
        fig, ax = plt.subplots()

        plt.imshow(img)
        plt.xlim(-1 * (width / 10.0), width + 1 * (width / 10.0))
        plt.ylim(-1 * (height / 10.0), height + 1 * (height / 10.0))
        plt.gca().invert_yaxis()

    w_crops = get_crops_parameters(width, crop, over, scale)
    h_crops = get_crops_parameters(height, crop, over, scale)
    N = len(w_crops) * len(h_crops)

    if not save_visualization:
        print(str((frame_name))+", ", end='', flush=True)
    else:
        print(str((frame_name)) + " ("+str(N)+" crops per frame), ", end='', flush=True)

    #print ("Number of crops:", N)

    crops = []
    i = 0
    for w_crop in w_crops:
        for h_crop in h_crops:
            if show or save_visualization:
                jitter = random.uniform(0, 1) * 15

                ax.add_patch(
                    patches.Rectangle(
                        (w_crop[0] + jitter, h_crop[0] + jitter),
                        scale * crop,
                        scale * crop, fill=False, linewidth=2.0, color=np.random.rand(3, 1)  # color=cmap(i)
                    )
                )

            area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + scale * crop), int(h_crop[0] + scale * crop))
            cropped_img = img.crop(box=area)
            cropped_img = cropped_img.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_img.load()

            file_name = frame_name + "/" + str(i).zfill(4) + ".jpg"
            if save_crops:
                cropped_img.save(out_folder + file_name)
            i += 1

            crops.append((file_name, area))
    if show:
        plt.show()
    if save_visualization:
        plt.savefig(viz_path+'crops_viz.png')
        cropped_img.save(viz_path + '_sample_first_crop.jpg')

    return crops

def crop_from_one_frame_WITH_MASK(frame_path, out_folder, crop, over, scale, show, save_crops=True, save_visualization=True, mask_url='', viz_path=''):
    # crop*scale is the size inside input image
    # crop is the size of output image
    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]

    if not os.path.exists(out_folder+frame_name+"/"):
        os.makedirs(out_folder+frame_name+"/")

    img = Image.open(frame_path)
    mask = Image.open(mask_url)

    if mask.mode is not "L":
        mask = mask.convert("L")

    width, height = img.size

    if show or save_visualization:
        fig, ax = plt.subplots()

        plt.imshow(img)

        plt.imshow(mask,alpha=0.4)

        plt.xlim(-1 * (width / 10.0), width + 1 * (width / 10.0))
        plt.ylim(-1 * (height / 10.0), height + 1 * (height / 10.0))
        plt.gca().invert_yaxis()



    w_crops = get_crops_parameters(width, crop, over, scale)
    h_crops = get_crops_parameters(height, crop, over, scale)
    N = len(w_crops) * len(h_crops)

    if not save_visualization:
        print(str((frame_name))+", ", end='', flush=True)
    else:
        print(str((frame_name)) + " ("+str(N)+" crops per frame), ", end='', flush=True)

    crops = []
    i = 0
    for w_crop in w_crops:
        for h_crop in h_crops:
            area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + scale * crop), int(h_crop[0] + scale * crop))
            cropped_img = img.crop(box=area)
            cropped_img = cropped_img.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_img.load()

            cropped_mask = mask.crop(box=area)
            cropped_mask = cropped_mask.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_mask.load()

            # four corners
            a = cropped_mask.crop(box=(0, 0, crop*(1-over), crop*(1-over)))
            b = cropped_mask.crop(box=(0, crop * (over), crop * (1 - over), crop * (1 - over)+ crop * (over)))
            c = cropped_mask.crop(box=(crop * over, 0, crop * (1 - over) + crop*over, crop * (1 - over)))
            d = cropped_mask.crop(box=(crop * over, crop * over, crop * (1 - over) + crop * over, crop * (1 - over)+crop * over))

            corner_empty = False
            for p in [a,b,c,d]:
                p.load()
                lum = np.sum(np.sum(p.getextrema(), 0))
                #print(p.size, lum)
                if lum == 0:
                    corner_empty = True
                    break

            if corner_empty:
                continue

            extrema = cropped_mask.getextrema()
            extrema_sum = np.sum(extrema,0)
            #print("summed extrema", extrema_sum)

            if extrema_sum == 0: # and extrema_sum[1] == 0:
                continue

            if show or save_visualization:
                jitter = random.uniform(0, 1) * 15

                ax.add_patch(
                    patches.Rectangle(
                        (w_crop[0] + jitter, h_crop[0] + jitter),
                        scale * crop,
                        scale * crop, fill=False, linewidth=2.0, color=np.random.rand(3, 1)  # color=cmap(i)
                    )
                )
            file_name = frame_name + "/" + str(i).zfill(4) + ".jpg"
            if save_crops:
                cropped_img.save(out_folder + file_name)

            i += 1

            crops.append((file_name, area))
    if show:
        plt.show()
    if save_visualization:
        plt.savefig(viz_path+'crops_viz.png')
        cropped_img.save(viz_path + '_sample_first_crop.jpg')

    return crops

def mask_from_one_frame(frame_path, SETTINGS, mask_folder):
    frame_image = Image.open(frame_path)

    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]
    if not os.path.exists(mask_folder+frame_name+"/"):
        os.makedirs(mask_folder+frame_name+"/")

    print(str((frame_name))+", ", end='', flush=True)

    ow, oh = frame_image.size

    # now we want to resize the image so we have height equal to crop size - create only one row of crops
    # we will have to reproject back the scales
    crop = SETTINGS["attention_crop"]

    nh = crop
    scale_full_img = nh / oh
    nw = ow * scale_full_img
    over = SETTINGS["attention_over"]

    tmp = frame_image.resize((int(nw), int(nh)), Image.ANTIALIAS)
    mask_crops = crop_from_one_img(tmp, crop, over, 1.0, folder_name=mask_folder, frame_name=frame_name+"/")

    return mask_crops, scale_full_img

