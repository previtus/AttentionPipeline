# Server tricks with matplotlib plotting
import matplotlib, os
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

# input frames images
# output marked frames images

def main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS):
    import os
    import numpy as np
    from crop_functions import crop_from_one_frame, crop_from_one_frame_WITH_MASK
    from yolo_handler import run_yolo
    from mark_frame_with_bbox import annotate_image_with_bounding_boxes
    from visualize_time_measurement import visualize_time_measurements
    from nms import non_max_suppression_fast,non_max_suppression_tf
    from pathlib import Path

    video_file_root_folder = str(Path(INPUT_FRAMES).parents[1])
    crops_folder = video_file_root_folder + "/temporary"+RUN_NAME+"/crops/"
    output_frames_folder = video_file_root_folder + "/output"+RUN_NAME+"/frames/"
    output_measurement_viz = video_file_root_folder + "/output"+RUN_NAME+"/graphs"
    if not os.path.exists(crops_folder):
        os.makedirs(crops_folder)
    if not os.path.exists(output_frames_folder):
        os.makedirs(output_frames_folder)

    attention_model = SETTINGS["attention"]

    # Frames to crops
    print("################## Cropping frames ##################")
    crop_per_frames = []
    crop_number_per_frames = []
    frame_files = sorted(os.listdir(INPUT_FRAMES))
    print("##",len(frame_files),"of frames")

    mask = video_file_root_folder+'/mask.jpg'

    for i in range(0, len(frame_files)):
        frame_path = INPUT_FRAMES + frame_files[i]

        if attention_model:
            crops = crop_from_one_frame_WITH_MASK(frame_path, crops_folder, SETTINGS["crop"], SETTINGS["over"], SETTINGS["scale"], show=False, save=True, mask_url=mask)
        else:
            crops = crop_from_one_frame(frame_path, crops_folder, SETTINGS["crop"], SETTINGS["over"], SETTINGS["scale"], show=False, save=True)

        crop_per_frames.append(crops)
        crop_number_per_frames.append(len(crops))

    # Run YOLO on crops
    print("")
    print("################## Running Model ##################")

    tmp = video_file_root_folder + "/temporary"+RUN_NAME+"/tmp/"

    evaluation_times, bboxes_per_frames = run_yolo(crops_folder, tmp, crop_number_per_frames, crop_per_frames, SETTINGS["scale"], SETTINGS["crop"])
    num_frames = len(crop_number_per_frames)

    #bboxes_per_frames = sort_out_crop_coords_and_bboxes(crop_per_frames, bboxes)

    #print (len(bboxes_per_frames), bboxes_per_frames)
    #print (len(bboxes_per_frames[0]), bboxes_per_frames[0])
    #print (len(bboxes_per_frames[0][0]), bboxes_per_frames[0][0])

    print("################## Annotating frames ##################")

    iou_threshold = 0.5
    limit_prob_lowest = 0 #0.70 # inside we limited for 0.3

    import tensorflow as tf
    sess = tf.Session()
    for i in range(0,len(frame_files)):
        test_bboxes = bboxes_per_frames[i]

        arrays = []
        scores = []
        for j in range(0,len(test_bboxes)):
            if test_bboxes[j][0] == 'person':
                score = test_bboxes[j][2]
                if score > limit_prob_lowest:
                    arrays.append(list(test_bboxes[j][1]))
                    scores.append(score)
        arrays = np.array(arrays)

        if len(arrays) == 0:
            continue

        person_id = 0

        DEBUG_TURN_OFF_NMS = False
        if not DEBUG_TURN_OFF_NMS:
            nms_arrays = non_max_suppression_fast(arrays, iou_threshold)
            reduced_bboxes_1 = []
            for j in range(0,len(nms_arrays)):
                a = ['person',nms_arrays[j],0.0,person_id]
                reduced_bboxes_1.append(a)

            print("test_bboxes .shape (with debug crop bboxes)",np.array(test_bboxes).shape)
            print("arrays .shape",np.array(arrays).shape)

            nms_arrays, scores = non_max_suppression_tf(sess, arrays,scores,50,iou_threshold)
            reduced_bboxes_2 = []
            for j in range(0,len(nms_arrays)):
                a = ['person',nms_arrays[j],scores[j],person_id]
                reduced_bboxes_2.append(a)

            test_bboxes = reduced_bboxes_2

        annotate_image_with_bounding_boxes(INPUT_FRAMES + frame_files[i], output_frames_folder + frame_files[i], test_bboxes,
                                           draw_text=False, save=True, show=False)
    sess.close()
    print (len(evaluation_times),evaluation_times)

    evaluation_times[0] = evaluation_times[1] # ignore first large value
    visualize_time_measurements([evaluation_times], ["Evaluation"], "Time measurements all frames", show=False, save=True, save_path=output_measurement_viz+'_1.png')

    # crop_number_per_frames
    last = 0
    summed_frame_measurements = []
    for f in range(0,num_frames):
        till = crop_number_per_frames[f]
        sub = evaluation_times[last:last+till]
        summed_frame_measurements.append(sum(sub))
        #print(last,till,sum(sub))

        last = till

    visualize_time_measurements([summed_frame_measurements], ['time per frame'], "Time measurements per frame",xlabel='frame #',
                                show=False, save=True, save_path=output_measurement_viz+'_3.png')

    import shutil
    temp_dir_del = video_file_root_folder + "/temporary" + RUN_NAME
    if os.path.exists(temp_dir_del):
        shutil.rmtree(temp_dir_del)

"""
INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/"
SETTINGS = {}
SETTINGS["crop"] = 544 ## crop_sizes_possible = [288,352,416,480,544] # multiples of 32
SETTINGS["over"] = 0.6
SETTINGS["scale"] = 1.0
RUN_NAME = "_Test"

main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS)


INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/bag exchange/input/frames/"
SETTINGS = {}
SETTINGS["crop"] = 1024 ## crop_sizes_possible = [288,352,416,480,544] # multiples of 32
SETTINGS["over"] = 0.6
SETTINGS["scale"] = 1.0
RUN_NAME = "_Test"

main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS)
"""

from datetime import *

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Find BBoxes in video.')
parser.add_argument('-crop', help='size of crops, enter multiples of 32', default='544')
parser.add_argument('-over', help='percentage of overlap, 0-1', default='0.6')
parser.add_argument('-scale', help='additional undersampling', default='1.0')
parser.add_argument('-input', help='path to folder full of frame images',
                    default="/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/")
parser.add_argument('-name', help='run name - will output in this dir', default='_Test-'+day+month)
parser.add_argument('-attention', help='use guidance of attention', default='True')

if __name__ == '__main__':
    args = parser.parse_args()

    INPUT_FRAMES = args.input
    SETTINGS = {}
    SETTINGS["crop"] = float(args.crop)  ## crop_sizes_possible = [288,352,416,480,544] # multiples of 32
    SETTINGS["over"] = float(args.over)
    SETTINGS["scale"] = float(args.scale)
    SETTINGS["attention"] = (args.attention == 'True')
    RUN_NAME = args.name

    #INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/DrivingNY/input/frames_1fps_1174_sub/"
    #SETTINGS["attention"] = False
    #RUN_NAME = "nydrivetest"

    #Test this:
    #RUN_NAME += "b"
    #SETTINGS["attention"] = False

    print(RUN_NAME, SETTINGS)

    main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS)

