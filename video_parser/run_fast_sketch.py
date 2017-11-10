# Server tricks with matplotlib plotting
import matplotlib, os
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

# input frames images
# output marked frames images

def main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS):
    import os
    from shutil import copyfile

    import numpy as np
    from crop_functions import crop_from_one_frame, crop_from_one_frame_WITH_MASK, mask_from_one_frame
    from yolo_handler import run_yolo
    from mark_frame_with_bbox import annotate_image_with_bounding_boxes, mask_from_evaluated_bboxes
    from visualize_time_measurement import visualize_time_measurements
    from nms import non_max_suppression_fast,non_max_suppression_tf
    from data_handler import save_string_to_file
    from pathlib import Path
    from timeit import default_timer as timer

    video_file_root_folder = str(Path(INPUT_FRAMES).parents[1])
    mask_folder = video_file_root_folder + "/temporary"+RUN_NAME+"/masks/"
    mask_crop_folder = video_file_root_folder + "/temporary"+RUN_NAME+"/mask_crops/"
    crops_folder = video_file_root_folder + "/temporary"+RUN_NAME+"/crops/"
    output_frames_folder = video_file_root_folder + "/output"+RUN_NAME+"/frames/"
    output_measurement_viz = video_file_root_folder + "/output"+RUN_NAME+"/graphs"
    for folder in [crops_folder, mask_folder, output_frames_folder, mask_crop_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    attention_model = SETTINGS["attention"]

    # Frames to crops
    frame_files = sorted(os.listdir(INPUT_FRAMES))

    print("################## Mask generation ##################")
    mask_names = []

    summed_mask_croping_time = []

    if attention_model:
        print("##", len(frame_files), "of frames")

        # 1 generate crops from full images

        mask_crops_per_frames = []
        scales_per_frames = []
        mask_crops_number_per_frames = []
        for i in range(0, len(frame_files)):
            start = timer()

            frame_path = INPUT_FRAMES + frame_files[i]
            mask_crops, scale_full_img = mask_from_one_frame(frame_path, SETTINGS, mask_crop_folder)
            mask_crops_per_frames.append(mask_crops)
            mask_crops_number_per_frames.append(len(mask_crops))
            scales_per_frames.append(scale_full_img)

            end = timer()
            time = (end - start)
            summed_mask_croping_time.append(time)

            # input img: frame_path
            # scale stuff, eval bboxes, make mask, save temp
            # masks[] = ...
        print("")
        #print(len(mask_crops_per_frames), mask_crops_per_frames)
        #print(len(scales_per_frames), scales_per_frames)

        # 2 eval these
        mask_tmp = video_file_root_folder + "/temporary" + RUN_NAME + "/mask_tmp/"
        masks_evaluation_times, masks_additional_times, bboxes_per_frames = run_yolo(mask_crop_folder, mask_tmp, mask_crops_number_per_frames, mask_crops_per_frames,1.0,SETTINGS["attention_crop"], INPUT_FRAMES,frame_files,resize_frames=scales_per_frames, VERBOSE=0)
        #print("bboxes_per_frames", len(bboxes_per_frames), bboxes_per_frames )
        #print("mask evaluation", len(masks_evaluation_times), masks_evaluation_times )

        # 3 make mask images accordingly
        for frame_i in range(0,len(bboxes_per_frames)):
            start = timer()

            bboxes = bboxes_per_frames[frame_i]
            scale = scales_per_frames[frame_i]
            mask_from_evaluated_bboxes(INPUT_FRAMES + frame_files[frame_i], mask_folder + frame_files[frame_i], bboxes, scale, SETTINGS["extend_mask_by"])
            mask_names.append(mask_folder + frame_files[frame_i])

            end = timer()
            time = (end - start)
            summed_mask_croping_time[frame_i] += time

        #print(mask_names)

    print("################## Cropping frames ##################")
    print("##",len(frame_files),"of frames")
    crop_per_frames = []
    crop_number_per_frames = []
    summed_croping_time = []

    save_one_crop_vis = True
    for i in range(0, len(frame_files)):
        start = timer()

        frame_path = INPUT_FRAMES + frame_files[i]

        if attention_model:
            mask = mask_names[i]
            crops = crop_from_one_frame_WITH_MASK(frame_path, crops_folder, SETTINGS["crop"], SETTINGS["over"], SETTINGS["scale"], show=False, save_crops=False, save_visualization=save_one_crop_vis, mask_url=mask, viz_path=output_measurement_viz)
        else:
            crops = crop_from_one_frame(frame_path, crops_folder, SETTINGS["crop"], SETTINGS["over"], SETTINGS["scale"], show=False, save_visualization=save_one_crop_vis, save_crops=False, viz_path=output_measurement_viz)

        crop_per_frames.append(crops)
        crop_number_per_frames.append(len(crops))
        save_one_crop_vis = False

        end = timer()
        time = (end - start)
        summed_croping_time.append(time)


    tmp_crops = crop_from_one_frame(INPUT_FRAMES + frame_files[0], crops_folder, SETTINGS["crop"], SETTINGS["over"], SETTINGS["scale"],
                                show=False, save_visualization=False, save_crops=False,viz_path='')
    max_number_of_crops_per_frame = len(tmp_crops)

    #print("crop_per_frames ", len(crop_per_frames), crop_per_frames)
    #print("crop_number_per_frames ", len(crop_number_per_frames), crop_number_per_frames)

    # Run YOLO on crops
    print("")
    print("################## Running Model ##################")

    tmp = video_file_root_folder + "/temporary"+RUN_NAME+"/tmp/"

    pureEval_times, ioPlusEval_times, bboxes_per_frames = run_yolo(crops_folder, tmp, crop_number_per_frames, crop_per_frames, SETTINGS["scale"], SETTINGS["crop"], INPUT_FRAMES,frame_files)
    num_frames = len(crop_number_per_frames)
    num_crops = len(crop_per_frames[0])

    #bboxes_per_frames = sort_out_crop_coords_and_bboxes(crop_per_frames, bboxes)

    #print (len(bboxes_per_frames), bboxes_per_frames)
    #print (len(bboxes_per_frames[0]), bboxes_per_frames[0])
    #print (len(bboxes_per_frames[0][0]), bboxes_per_frames[0][0])

    print("################## Annotating frames ##################")

    iou_threshold = 0.5
    limit_prob_lowest = 0 #0.70 # inside we limited for 0.3

    print_first = True

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
            # no bboxes found in there, still we should copy the frame img
            copyfile(INPUT_FRAMES + frame_files[i], output_frames_folder + frame_files[i])
            continue

        person_id = 0

        DEBUG_TURN_OFF_NMS = False
        if not DEBUG_TURN_OFF_NMS:
            #nms_arrays = non_max_suppression_fast(arrays, iou_threshold)
            #reduced_bboxes_1 = []
            #for j in range(0,len(nms_arrays)):
            #    a = ['person',nms_arrays[j],0.0,person_id]
            #    reduced_bboxes_1.append(a)

            nms_arrays, scores = non_max_suppression_tf(sess, arrays,scores,50,iou_threshold)
            reduced_bboxes_2 = []
            for j in range(0,len(nms_arrays)):
                a = ['person',nms_arrays[j],scores[j],person_id]
                reduced_bboxes_2.append(a)

            test_bboxes = reduced_bboxes_2

        if print_first:
            print("Annotating with bboxes of len: ", len(test_bboxes) ,"files in:", INPUT_FRAMES + frame_files[i], ", out:", output_frames_folder + frame_files[i])
            print_first = False
        annotate_image_with_bounding_boxes(INPUT_FRAMES + frame_files[i], output_frames_folder + frame_files[i], test_bboxes,
                                           draw_text=False, save=True, show=False, thickness=SETTINGS["thickness"])

    sess.close()
    print (len(pureEval_times),pureEval_times)

    #evaluation_times[0] = evaluation_times[1] # ignore first large value
    #masks_evaluation_times[0] = masks_evaluation_times[1] # ignore first large value
    visualize_time_measurements([pureEval_times], ["Evaluation"], "Time measurements all frames", show=False, save=True, save_path=output_measurement_viz+'_1.png',  y_min=0.0, y_max=0.5)
    visualize_time_measurements([pureEval_times], ["Evaluation"], "Time measurements all frames", show=False, save=True, save_path=output_measurement_viz+'_1.png',  y_min=0.0, y_max=0.0)

    # crop_number_per_frames
    last = 0
    summed_frame_measurements = []
    for f in range(0,num_frames):
        till = crop_number_per_frames[f]
        sub = pureEval_times[last:last+till]
        summed_frame_measurements.append(sum(sub))
        #print(last,till,sum(sub))
        last = till

    if attention_model:
        last = 0
        summed_mask_measurements = []
        for f in range(0,num_frames):
            till = mask_crops_number_per_frames[f]
            sub = masks_evaluation_times[last:last+till]
            summed_mask_measurements.append(sum(sub))
            #print(last,till,sum(sub))
            last = till

    avg_time_crop = np.mean(pureEval_times[1:])
    max_time_per_frame_estimate = max_number_of_crops_per_frame * avg_time_crop
    estimated_max_time_per_frame = [max_time_per_frame_estimate] * num_frames

    if attention_model:
        arrs = [summed_frame_measurements, summed_mask_measurements, summed_croping_time, summed_mask_croping_time,
                ioPlusEval_times, masks_additional_times, estimated_max_time_per_frame]
        names = ['image eval', 'mask eval', 'cropping image', 'cropping mask', 'image eval+io', 'mask eval+io', 'estimated max']
    else:
        arrs = [summed_frame_measurements, summed_croping_time, ioPlusEval_times]
        names = ['image eval','cropping image', 'image eval+io']

    visualize_time_measurements(arrs, names, "Time measurements per frame",xlabel='frame #',
                                show=False, save=True, save_path=output_measurement_viz+'_3.png')

    # save settings
    avg_time_frame = np.mean(summed_frame_measurements[1:])
    strings = [RUN_NAME+" "+str(SETTINGS), INPUT_FRAMES, str(num_crops)+" crops per frame * "+ str(num_frames) + " frames", "Time:" + str(avg_time_crop) + " avg per crop, " + str(avg_time_frame) + " avg per frame."]
    save_string_to_file(strings, output_measurement_viz+'_settings.txt')

    keep_temporary = True
    if not keep_temporary:
        import shutil
        temp_dir_del = video_file_root_folder + "/temporary" + RUN_NAME
        if os.path.exists(temp_dir_del):
            shutil.rmtree(temp_dir_del)


from datetime import *

months = ["unk","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month = (months[datetime.now().month])
day = str(datetime.now().day)

import argparse

parser = argparse.ArgumentParser(description='Project: Find BBoxes in video.')
parser.add_argument('-crop', help='size of crops, enter multiples of 32', default='544')
parser.add_argument('-over', help='percentage of overlap, 0-1', default='0.6')
parser.add_argument('-attcrop', help='size of crops for attention model', default='608')
parser.add_argument('-attover', help='percentage of overlap for attention model', default='0.65')
parser.add_argument('-scale', help='additional undersampling', default='1.0')
parser.add_argument('-input', help='path to folder full of frame images',
                    default="/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/")
parser.add_argument('-name', help='run name - will output in this dir', default='_Test-'+day+month)
parser.add_argument('-attention', help='use guidance of automatic attention model', default='True')
parser.add_argument('-thickness', help='thickness', default='10,2')
parser.add_argument('-extendmask', help='extend mask by', default='300')

if __name__ == '__main__':
    args = parser.parse_args()

    INPUT_FRAMES = args.input
    RUN_NAME = args.name
    SETTINGS = {}
    SETTINGS["attention_crop"] = float(args.attcrop)
    SETTINGS["attention_over"] = float(args.attover)
    SETTINGS["crop"] = float(args.crop)  ## crop_sizes_possible = [288,352,416,480,544] # multiples of 32
    SETTINGS["over"] = float(args.over)
    SETTINGS["scale"] = float(args.scale)
    SETTINGS["attention"] = (args.attention == 'True')
    SETTINGS["extend_mask_by"] = int(args.extendmask)
    thickness = str(args.thickness).split(",")
    SETTINGS["thickness"] = [float(thickness[0]), float(thickness[1])]

    SETTINGS["crop"] = 1000
    SETTINGS["over"] = 0.65
    INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/"
    RUN_NAME = "_testWithoutSavingCrops_"+day+month

    print(RUN_NAME, SETTINGS)
    main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS)

