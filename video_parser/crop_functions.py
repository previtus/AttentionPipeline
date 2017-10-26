import os
import numpy, random
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

def crop_from_one_frame(frame_path, out_folder, crop, over, scale, show, save=True):
    # crop*scale is the size inside input image
    # crop is the size of output image
    frame_name = os.path.basename(frame_path)
    frame_name = frame_name[0:-4]

    if not os.path.exists(out_folder+frame_name+"/"):
        os.makedirs(out_folder+frame_name+"/")

    print(str(int(frame_name))+", ", end='', flush=True)
    #print (frame_name, out_folder+frame_name+"/")

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

    #print ("Number of crops:", N)

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
                        scale * crop, fill=False, linewidth=2.0, color=numpy.random.rand(3, 1)  # color=cmap(i)
                    )
                )

            area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + scale * crop), int(h_crop[0] + scale * crop))
            cropped_img = img.crop(box=area)
            cropped_img = cropped_img.resize((crop, crop), resample=Image.ANTIALIAS)
            cropped_img.load()

            file_name = out_folder + frame_name + "/" + str(i).zfill(4) + ".jpg"
            if save:
                cropped_img.save(file_name)
            i += 1

            crops.append((file_name, area))
    if show:
        plt.show()
    return crops

def trim(img, border):
    bg = Image.new(img.mode, img.size, border)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)

def combine_crop_images_2(crop_folder, crop, over, scale, resize_output=1.0):
    image_files = sorted(os.listdir(crop_folder))
    img = Image.open(crop_folder+image_files[0])
    dimension = img.size[0]
    print (dimension)

    width, height = (2000,1500)
    img = Image.new(img.mode, (width, height))

    w_crops = get_crops_parameters(width, crop, over, scale)
    h_crops = get_crops_parameters(height, crop, over, scale)
    N = len(w_crops) * len(h_crops)

    print ("Number of crops:", N)

    i = 0
    for w_crop in w_crops:
        for h_crop in h_crops:
            area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + scale * crop), int(h_crop[0] + scale * crop))
            crop_img = Image.open(crop_folder + image_files[i])

            tmp_img = Image.new(img.mode, (width, height))
            tmp_img.paste(crop_img, area)

            img = ImageChops.lighter(img, tmp_img)

            i += 1

    img = trim(img,(0,0,0))

    if resize_output is not 1.0:
        img = img.resize((int(resize_output*img.size[0]), int(resize_output*img.size[1])))
    return img


def combine_crop_images(crop_folder, h_multiples, w_multiples, overlap):
    image_files = sorted(os.listdir(crop_folder))
    print(len(image_files))

    img = Image.open(crop_folder+image_files[0])
    dimension = img.size[0]
    print (dimension)

    new_img = Image.new("RGBA", (w_multiples*dimension, h_multiples*dimension), color=(0, 0, 0, 0))

    i = 0
    for w in range(0, w_multiples):
        for h in range(0,h_multiples):
            print (h, w, i, (w*dimension, h*dimension))
            img = Image.open(crop_folder + image_files[i])

            #new_img.paste(img, (w*dimension-(dimension*overlap), h*dimension-(dimension*overlap), (w+1)*dimension, (h+1)*dimension))
            shift = int(dimension*(overlap))
            shift_w = w*shift
            shift_h = h*shift
            #tmp_img = Image.new("RGBA", (w_multiples * dimension, h_multiples * dimension), color=(0, 0, 0, 0))
            #tmp_img.paste(img, (w * dimension - shift_w, h * dimension - shift_h))

            #new_img = Image.blend(new_img, tmp_img, 0.5)

            new_img.paste(img, (w * dimension - shift_w, h * dimension - shift_h))


            i += 1

    scale = 0.5
    new_img = new_img.resize((int(scale*new_img.size[0]), int(scale*new_img.size[1])))
    new_img.show()

def dump_frames_to_folder(input_folder, output, crop_size, over, output_resize, limit=None):
    frame_dirs = sorted(os.listdir(input_folder))
    if limit is not None:
        frame_dirs = frame_dirs[0:limit]
    if not os.path.exists(output):
        os.makedirs(output)

    i = 0
    for frame_dir in frame_dirs:
        crop_folder = input_folder+frame_dir+"/"
        img = combine_crop_images_2(crop_folder, crop_size, over, output_resize)

        img.save(output + str(i).zfill(4) + ".jpg")
        i+=1

'''all combined crops to folder'''
#input_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set3_1024_0.6/"
#output = "/home/ekmek/intership_project/video_parser/PL_Pizza/VIDTMP_set3_1024_0.6/"
#dump_frames_to_folder(input_folder, output, 1024, 0.6, 1.0)

# ffmpeg -r 6/1 -i VIDTMP_set2_288_0.6/%04d.jpg -c:v libx264 -vf scale=trunc(iw/2)*2:trunc(ih/2)*2 fps=6 -pix_fmt yuv420p VIDTMP_set2_288_0.6_out.mp4
# ffmpeg -r 6/1 -i VIDTMP_set1_544_0.6/%04d.jpg -c:v libx264 -vf fps=6 -pix_fmt yuv420p VIDTMP_set1_544_0.6_out.mp4

"""
input_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set2_288_0.6/"
output = "/home/ekmek/intership_project/video_parser/PL_Pizza/VIDTMP_set2_288_0.6/"
dump_frames_to_folder(input_folder, output, 288, 0.6, 1.0)

input_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set1_544_0.6/"
output = "/home/ekmek/intership_project/video_parser/PL_Pizza/VIDTMP_set1_544_0.6/"
dump_frames_to_folder(input_folder, output, 544, 0.6, 1.0)
"""

'''1 crop to img'''
#crop_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set3_1024_0.6/0000_marked/"
#combine_crop_images(crop_folder, 2, 3, 0.6)
#img = combine_crop_images_2(crop_folder, 1024, 0.6, 1.0)
#img.show()

#crop_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set2_288_0.6/0000/"
#img = combine_crop_images_2(crop_folder, 288, 0.6, 1.0)
#img.show()
