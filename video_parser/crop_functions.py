import os
import numpy as np
import random
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from squares_filling import get_crops_parameters, best_squares_overlap

def crop_from_one_img(img, horizontal_splits, overlap_px, scale, show=False, save_crops=True, folder_name='', frame_name=''):

    width, height = img.size

    if show:
        fig, ax = plt.subplots()

        plt.imshow(img)
        plt.xlim(-1 * (width / 10.0), width + 1 * (width / 10.0))
        plt.ylim(-1 * (height / 10.0), height + 1 * (height / 10.0))
        plt.gca().invert_yaxis()

    column_list, row_list = best_squares_overlap(width,height,horizontal_splits,overlap_px)
    crop = column_list[0][1] - column_list[0][0]
    w_crops = column_list
    h_crops = row_list
    #print("after w",w_crops)
    #print("after h",h_crops)

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
                if not os.path.exists(folder_name + frame_name + "/"):
                    os.makedirs(folder_name + frame_name + "/")
                cropped_img.save(folder_name + file_name)
                crops.append((file_name, area))
            else:
                crops.append((None,area))
            i += 1

    if show:
        plt.show()

    return crops, crop

def get_number_of_crops_from_frame(frame_path, horizontal_splits,overlap_px):
    # crop*scale is the size inside input image
    # crop is the size of output image

    img = Image.open(frame_path)
    width, height = img.size

    column_list, row_list = best_squares_overlap(width,height,horizontal_splits,overlap_px)
    w_crops = column_list
    h_crops = row_list

    N = len(w_crops) * len(h_crops)
    return N

def crop_from_one_frame(frame_path, out_folder, horizontal_splits,overlap_px, show, save_crops=True, save_visualization=True, viz_path=''):
    # crop*scale is the size inside input image
    # crop is the size of output image

    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]

    if save_crops:
        if not os.path.exists(out_folder+frame_name+"/"):
            os.makedirs(out_folder+frame_name+"/")

    img = Image.open(frame_path)
    width, height = img.size

    if show or save_visualization:
        fig, ax = plt.subplots()

        plt.imshow(img)
        plt.xlim(-1 * (width / 10.0), width + 1 * (width / 10.0))
        plt.ylim(-1 * (height / 10.0), height + 1 * (height / 10.0))
        plt.gca().invert_yaxis()

    #w_crops = get_crops_parameters(width, crop, over, scale)
    #h_crops = get_crops_parameters(height, crop, over, scale)
    column_list, row_list = best_squares_overlap(width,height,horizontal_splits,overlap_px)
    crop = column_list[0][1] - column_list[0][0]
    w_crops = column_list
    h_crops = row_list

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
                        crop,
                        crop, fill=False, linewidth=2.0, color=np.random.rand(3, 1)  # color=cmap(i)
                    )
                )

            area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + crop), int(h_crop[0] + crop))
            cropped_img = img.crop(box=area)
            cropped_img = cropped_img.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_img.load()

            file_name = frame_name + "/" + str(i).zfill(4) + ".jpg"
            if save_crops:
                cropped_img.save(out_folder + file_name)
                crops.append((file_name, area))
            else:
                crops.append((None, area))
            i += 1

    if show:
        plt.show()
    if save_visualization:
        plt.savefig(viz_path+'crops_viz.png')
        cropped_img.save(viz_path + '_sample_first_crop.jpg')

    return crops, crop

#@profile
def crop_from_one_frame_WITH_MASK_in_mem(img, mask, frame_path, out_folder, horizontal_splits, overlap_px, mask_over, show, save_crops=True, save_visualization=True, viz_path=''):
    # V3 - mask carried in memory

    # crop*scale is the size inside input image
    # crop is the size of output image
    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]

    if save_crops:
        if not os.path.exists(out_folder+frame_name+"/"):
            os.makedirs(out_folder+frame_name+"/")

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

    #w_crops = get_crops_parameters(width, crop, over, scale)
    #h_crops = get_crops_parameters(height, crop, over, scale)
    column_list, row_list = best_squares_overlap(width,height,horizontal_splits,overlap_px)
    crop = column_list[0][1] - column_list[0][0]
    w_crops = column_list
    h_crops = row_list

    N = len(w_crops) * len(h_crops)

    if not save_visualization:
        print(str((frame_name))+", ", end='', flush=True)
    else:
        print(str((frame_name)) + " ("+str(N)+" crops per frame), ", end='', flush=True)

    crops = []
    i = 0
    for w_crop in w_crops:
        for h_crop in h_crops:
            area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + crop), int(h_crop[0] + crop))
            cropped_img = img.crop(box=area)
            cropped_img = cropped_img.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_img.load()

            cropped_mask = mask.crop(box=area)
            cropped_mask = cropped_mask.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_mask.load()

            # four corners
            a = cropped_mask.crop(box=(0, 0, crop * (1 - mask_over), crop * (1 - mask_over)))
            b = cropped_mask.crop(box=(0, crop * (mask_over), crop * (1 - mask_over), crop * (1 - mask_over) + crop * (mask_over)))
            c = cropped_mask.crop(box=(crop * mask_over, 0, crop * (1 - mask_over) + crop * mask_over, crop * (1 - mask_over)))
            d = cropped_mask.crop(box=(crop * mask_over, crop * mask_over, crop * (1 - mask_over) + crop * mask_over, crop * (1 - mask_over) + crop * mask_over))

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
                        crop,
                        crop, fill=False, linewidth=2.0, color=np.random.rand(3, 1)[:,0]  # color=cmap(i)
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

    return crops, crop

def crop_from_one_frame_WITH_MASK(frame_path, out_folder, crop, over, scale, show, save_crops=True, save_visualization=True, mask_url='', viz_path=''):
    # V2 - mask held in another file

    # crop*scale is the size inside input image
    # crop is the size of output image
    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]

    if save_crops:
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

#@profile
def mask_from_one_frame(frame_path, SETTINGS, mask_folder):
    ## del attention_crop, attention_over,
    ## instead attention_h_num


    frame_image = Image.open(frame_path)

    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]
    #if not os.path.exists(mask_folder+frame_name+"/"):
    #    os.makedirs(mask_folder+frame_name+"/")

    print(str((frame_name))+", ", end='', flush=True)

    ow, oh = frame_image.size

    # now we want to resize the image so we have height equal to crop size - create only one row of crops
    # we will have to reproject back the scales

    horizontal_splits = SETTINGS["attention_horizontal_splits"]
    overlap_px = 0
    column_list, row_list = best_squares_overlap(ow,oh,horizontal_splits,overlap_px)
    crop = 608 * horizontal_splits

    #crop = 608 + (608 - overlap_px) * (horizontal_splits - 1)

    #crop = SETTINGS["attention_crop"]

    nh = crop
    scale_full_img = nh / oh
    nw = ow * scale_full_img

    tmp = frame_image.resize((int(nw), int(nh)), Image.ANTIALIAS)

    save_crops = True

    mask_crops, crop = crop_from_one_img(tmp, horizontal_splits, overlap_px, 1.0, folder_name=mask_folder, frame_name=frame_name+"/", save_crops=save_crops)

    return mask_crops, scale_full_img, crop

