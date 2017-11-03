import os
import random
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont


# CODE USED FROM YAD2K

def annotate_image_with_bounding_boxes(image_path, save_path, bboxes, ignore_crops_drawing=True, draw_text=True, show=False, save=True):
    #print(image_path, save_path)

    tmp_len = 80
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / tmp_len, 1., 1.)
                  for x in range(tmp_len)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.


    image = Image.open(image_path)

    for bbox in bboxes:
        #print (bbox)

        predicted_class = bbox[0]

        if predicted_class is 'crop' and ignore_crops_drawing:
            continue


        box = bbox[1]
        score = bbox[2]
        c = bbox[3]

        font = ImageFont.truetype(
                font=('font/FiraMono-Medium.otf'),
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        #thickness = (image.size[0] + image.size[1]) // 600
        thickness = int( 4 * score )

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])

        if draw_text:
            draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    #resize_output = 0.5
    #if resize_output is not 1.0:
    #    image = image.resize((int(resize_output*image.size[0]), int(resize_output*image.size[1])))

    if show:
        image.show()

    if save:
        image.save(save_path, quality=90)