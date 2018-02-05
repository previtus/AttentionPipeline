#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import colorsys
import imghdr
import os
import random
import sys

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head

from timeit import default_timer as timer


def _main(args, frames_paths, crops_bboxes, crop_value, resize_frames=None, verbose=1, person_only=True, allowed_number_of_boxes=100):
    '''

    :param args: yolo model args like in YAD2K
    :param frames_paths: list of paths to frame images
    :param crops_bboxes: list of lists - crops per frames
    :param verbose:
    :param person_only:
    :return:
    '''

    print("frames_paths", len(frames_paths), frames_paths)
    print("crops_bboxes", len(crops_bboxes), crops_bboxes)


    model_path = os.path.expanduser(args["model_path"])
    print(model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args["anchors_path"])
    classes_path = os.path.expanduser(args["classes_path"])

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    if verbose > 0:
        print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))


    ####### EVALUATION

    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args["score_threshold"],
        iou_threshold=args["iou_threshold"],
        max_boxes=allowed_number_of_boxes)

    pureEval_times = []
    ioPlusEval_times = []
    bboxes = []

    images_processed = 0
    evaluation_time = 0

    ### PROCESS FILES
    for frame_i in range(0,len(frames_paths)):
        start_loop = timer()

        images_processed += 1

        frame_path = frames_paths[frame_i]
        frame = Image.open(frame_path)

        if resize_frames is not None:
            resize_to = resize_frames[frame_i]

            ow, oh = frame.size

            nw = ow * resize_to
            nh = oh * resize_to

            frame = frame.resize((int(nw), int(nh)), Image.ANTIALIAS)

        crops_in_frame = crops_bboxes[frame_i]
        #print("Frame", frame_i, " with ", len(crops_in_frame), " crops.")
        sys.stdout.write("\rFrame " + str(frame_i) + " with " + str(len(crops_in_frame)) + " crops.")
        sys.stdout.flush()

        for crop_i in range(0, len(crops_in_frame)):
            crop = crops_in_frame[crop_i]
            area = crop[1]

            #area = (int(w_crop[0]), int(h_crop[0]), int(w_crop[0] + scale * crop), int(h_crop[0] + scale * crop))
            cropped_img = frame.crop(box=area)
            cropped_img = cropped_img.resize((int(crop_value), int(crop_value)), resample=Image.ANTIALIAS)
            cropped_img.load()

            image = cropped_img

            """
            image_data = np.array(image, dtype='float32')
            """
            if is_fixed_size:  # TODO: When resizing we can use minibatch input.
                resized_image = image.resize(
                    tuple(reversed(model_image_size)), Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')
            else:
                # Due to skip connection + max pooling in YOLO_v2, inputs must have
                # width and height as multiples of 32.
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
                resized_image = image.resize(new_image_size, Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')
                print(image_data.shape)


            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            if images_processed < 2:
                print("# image size: ",image_data.shape, image.size)

            ################# START #################
            start_eval = timer()
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            end_eval = timer()
            ################# END #################
            evaluation_time = (end_eval - start_eval)
            pureEval_times.append(evaluation_time)

            people = 0
            bboxes_image = []
            #print(num_frames, num_crops)
            for i, c in reversed(list(enumerate(out_classes))):

                predicted_class = class_names[c]

                if predicted_class == 'person':
                    people += 1
                if person_only and (predicted_class != 'person'):
                    continue

                box = out_boxes[i]
                score = out_scores[i]

                #print(predicted_class, box, score)

                bboxes_image.append([predicted_class, box, score, c])

            if verbose > 0:
                num = len(out_boxes)
                if person_only:
                    num = people
                print('Found {} boxes in crop {} of frame {} in {}s'.format(num, crop_i, frame_i, evaluation_time))


            bboxes.append(bboxes_image)

        end_loop = timer()
        loop_time = (end_loop - start_loop)
        ioPlusEval_times.append(loop_time - evaluation_time)

    #sess.close()

    return pureEval_times, ioPlusEval_times, bboxes

#if __name__ == '__main__':
#    print(parser.parse_args())
#    _main(parser.parse_args())
