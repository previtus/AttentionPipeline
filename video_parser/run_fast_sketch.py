# Server tricks with matplotlib plotting
import matplotlib, os, glob, fnmatch
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

from shutil import copyfile
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image

from crop_functions import crop_from_one_frame, mask_from_one_frame, crop_from_one_frame_WITH_MASK_in_mem, get_number_of_crops_from_frame
from yolo_handler import run_yolo
from mark_frame_with_bbox import annotate_image_with_bounding_boxes, mask_from_evaluated_bboxes, bboxes_to_mask, annotate_prepare
from visualize_time_measurement import visualize_time_measurements
from nms import py_cpu_nms, non_max_suppression_tf
from bbox_postprocessing import postprocess_bboxes_by_splitlines
from data_handler import save_string_to_file, saveDict, loadDict

# input frames images
# output marked frames images
#@profile
def main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS):
    video_file_root_folder = str(Path(INPUT_FRAMES).parents[1])
    output_frames_folder = video_file_root_folder + "/output/" + RUN_NAME + "/frames/"
    output_measurement_viz = video_file_root_folder + "/output/" + RUN_NAME + "/graphs"
    output_annotation = video_file_root_folder + "/output/" + RUN_NAME + "/annot"
    output_savedLastExp = video_file_root_folder + "/output/" + RUN_NAME + "/_lastExpiment.npy"

    mask_folder = video_file_root_folder + "/temporary/"+RUN_NAME+"/masks/"
    mask_crop_folder = video_file_root_folder + "/temporary/"+RUN_NAME+"/mask_crops/" # useless, but maybe for debug later
    crops_folder = video_file_root_folder + "/temporary/"+RUN_NAME+"/crops/" # also useless, but maybe for debug later
    for folder in [output_frames_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    attention_model = SETTINGS["attention"]
    attention_spread_frames = SETTINGS["att_frame_spread"]

    # Frames to crops
    files = sorted(os.listdir(INPUT_FRAMES))
    frame_files = fnmatch.filter(files, '*.jpg')
    annotation_files = fnmatch.filter(files, '*.xml')
    print("jpgs:",frame_files[0:2],"...","xmls:",annotation_files[0:2],"...")

    start_frame = SETTINGS["startframe"]
    frame_files = frame_files[start_frame:]
    allowed_number_of_boxes = SETTINGS["allowed_number_of_boxes"]

    if not SETTINGS["reuse_last_experiment"]:
        print("################## Mask generation ##################")

        summed_mask_croping_time = []

        if attention_model:
            print("##", len(frame_files), "of frames")

            # 1 generate crops from full images
            mask_crops_per_frames = []
            scales_per_frames = []
            mask_crops_number_per_frames = []
            for frame_i in range(0, len(frame_files)):
                start = timer()

                frame_path = INPUT_FRAMES + frame_files[frame_i]
                mask_crops, scale_full_img, attention_crop_TMP_SIZE_FOR_MODEL = mask_from_one_frame(frame_path,
                              SETTINGS, mask_crop_folder)  ### <<< mask_crops
                mask_crops_per_frames.append(mask_crops)
                mask_crops_number_per_frames.append(len(mask_crops))
                scales_per_frames.append(scale_full_img)

                end = timer()
                time = (end - start)
                summed_mask_croping_time.append(time)

            print("")

            # 2 eval these
            # calculate
            masks_evaluation_times, masks_additional_times, bboxes_per_frames = run_yolo(mask_crops_number_per_frames,
                      mask_crops_per_frames, attention_crop_TMP_SIZE_FOR_MODEL, INPUT_FRAMES, frame_files,
                      resize_frames=scales_per_frames, allowed_number_of_boxes=allowed_number_of_boxes,
                      VERBOSE=0, anchors_txt=SETTINGS["anchorfile"])

            # 3 make mask images accordingly
            save_masks = SETTINGS["debug_save_masks"]
            range_of_masks = []
            if save_masks is "one":
                range_of_masks = [0]
            elif save_masks is "all":
                range_of_masks = range(0, len(frame_files))

            for i in range_of_masks:
                # for i in range(0,len(frame_files)):
                print(output_measurement_viz + frame_files[i])
                tmp_mask_just_to_save_it_for_debug = mask_from_evaluated_bboxes(INPUT_FRAMES + frame_files[i],
                      output_measurement_viz + frame_files[i], bboxes_per_frames[i], scales_per_frames[i], 0)  # SETTINGS["extend_mask_by"]

        print("################## Cropping frames ##################")
        print("##", len(frame_files), "of frames")
        crop_per_frames = []
        crop_number_per_frames = []
        summed_croping_time = []

        save_one_crop_vis = True
        for frame_i in range(0, len(frame_files)):
            start = timer()

            frame_path = INPUT_FRAMES + frame_files[frame_i]

            if attention_model:

                if attention_spread_frames == 0:
                    bboxes = bboxes_per_frames[frame_i]
                    # print(len(bboxes), bboxes)

                else:
                    from_frame = max([frame_i - attention_spread_frames, 0])
                    to_frame = min([frame_i + attention_spread_frames, len(frame_files)]) + 1

                    bboxes = [item for sublist in bboxes_per_frames[from_frame:to_frame] for item in sublist]
                    # print(from_frame,"to",to_frame-1,len(bboxes), bboxes)

                scale = scales_per_frames[frame_i]
                img = Image.open(frame_path)
                mask = bboxes_to_mask(bboxes, img.size, scale, SETTINGS["extend_mask_by"])

                mask_over = 0.1  # SETTINGS["over"]
                horizontal_splits = SETTINGS["horizontal_splits"]
                overlap_px = SETTINGS["overlap_px"]
                crops, crop_TMP = crop_from_one_frame_WITH_MASK_in_mem(img, mask, frame_path, crops_folder,
                     horizontal_splits, overlap_px, mask_over, show=False, save_crops=False,
                     save_visualization=save_one_crop_vis, viz_path=output_measurement_viz)

            else:
                horizontal_splits = SETTINGS["horizontal_splits"]
                overlap_px = SETTINGS["overlap_px"]

                crops, crop_TMP = crop_from_one_frame(frame_path, crops_folder, horizontal_splits, overlap_px,
                      show=False, save_visualization=save_one_crop_vis, save_crops=False, viz_path=output_measurement_viz)

            crop_per_frames.append(crops)
            crop_number_per_frames.append(len(crops))
            save_one_crop_vis = False

            end = timer()
            time = (end - start)
            summed_croping_time.append(time)

        crop_TMP_SIZE_FOR_MODEL = crop_TMP

        horizontal_splits = SETTINGS["horizontal_splits"]
        overlap_px = SETTINGS["overlap_px"]

        max_number_of_crops_per_frame = get_number_of_crops_from_frame(INPUT_FRAMES + frame_files[0], horizontal_splits,
                                                                       overlap_px)
        # print("max_number_of_crops_per_frame",max_number_of_crops_per_frame)
        # tmp_crops,_ = crop_from_one_frame(INPUT_FRAMES + frame_files[0], crops_folder, horizontal_splits,overlap_px,
        #                            show=False, save_visualization=False, save_crops=False,viz_path='')
        # max_number_of_crops_per_frame = len(tmp_crops)

        # Run YOLO on crops
        print("")
        print("################## Running Model ##################")
        print("Crop size = ", crop_TMP_SIZE_FOR_MODEL)

        pureEval_times, ioPlusEval_times, bboxes_per_frames = run_yolo(crop_number_per_frames, crop_per_frames,
                                                                       crop_TMP_SIZE_FOR_MODEL, INPUT_FRAMES,
                                                                       frame_files, anchors_txt=SETTINGS["anchorfile"],
                                                                       allowed_number_of_boxes=allowed_number_of_boxes)
        num_frames = len(crop_number_per_frames)
        num_crops = len(crop_per_frames[0])

        print("################## Save Graphs ##################")

        print(len(pureEval_times), pureEval_times[0:3])

        # evaluation_times[0] = evaluation_times[1] # ignore first large value
        # masks_evaluation_times[0] = masks_evaluation_times[1] # ignore first large value
        visualize_time_measurements([pureEval_times], ["Evaluation"], "Time measurements all frames", show=False,
                                    save=True, save_path=output_measurement_viz + '_1.png', y_min=0.0, y_max=0.5)
        visualize_time_measurements([pureEval_times], ["Evaluation"], "Time measurements all frames", show=False,
                                    save=True, save_path=output_measurement_viz + '_1.png', y_min=0.0, y_max=0.0)

        last = 0
        summed_frame_measurements = []
        for f in range(0, num_frames):
            till = crop_number_per_frames[f]
            sub = pureEval_times[last:last + till]
            summed_frame_measurements.append(sum(sub))
            # print(last,till,sum(sub))
            last = till

        if attention_model:
            last = 0
            summed_mask_measurements = []
            for f in range(0, num_frames):
                till = mask_crops_number_per_frames[f]
                sub = masks_evaluation_times[last:last + till]
                summed_mask_measurements.append(sum(sub))
                # print(last,till,sum(sub))
                last = till

        avg_time_crop = np.mean(pureEval_times[1:])
        max_time_per_frame_estimate = max_number_of_crops_per_frame * avg_time_crop
        estimated_max_time_per_frame = [max_time_per_frame_estimate] * num_frames

        if attention_model:
            arrs = [summed_frame_measurements, summed_mask_measurements, summed_croping_time, summed_mask_croping_time,
                    ioPlusEval_times, masks_additional_times, estimated_max_time_per_frame]
            names = ['image eval', 'mask eval', 'cropping image', 'cropping mask', 'image eval+io', 'mask eval+io',
                     'estimated max']
        else:
            arrs = [summed_frame_measurements, summed_croping_time, ioPlusEval_times]
            names = ['image eval', 'cropping image', 'image eval+io']

        visualize_time_measurements(arrs, names, "Time measurements per frame", xlabel='frame #',
                                    show=False, save=True, save_path=output_measurement_viz + '_3.png')

        ## version b and c
        if attention_model:
            arrs = [summed_frame_measurements, summed_mask_measurements,
                    ioPlusEval_times, masks_additional_times, estimated_max_time_per_frame]
            names = ['image eval', 'mask eval', 'image eval+io', 'mask eval+io',
                     'estimated max']
            visualize_time_measurements(arrs, names, "Time measurements per frame", xlabel='frame #',
                                        show=False, save=True, save_path=output_measurement_viz + '_3b.png')

            arrs = [summed_frame_measurements, summed_mask_measurements, estimated_max_time_per_frame]
            names = ['image eval', 'mask eval', 'estimated max']
            visualize_time_measurements(arrs, names, "Time measurements per frame", xlabel='frame #',
                                        show=False, save=True, save_path=output_measurement_viz + '_3c.png')

        # save settings
        avg_time_frame = np.mean(summed_frame_measurements[1:])
        strings = [RUN_NAME + " " + str(SETTINGS), INPUT_FRAMES,
                   str(num_crops) + " crops per frame * " + str(num_frames) + " frames",
                   "Time:" + str(avg_time_crop) + " avg per crop, " + str(avg_time_frame) + " avg per frame.",
                   "Crop size in px was:" + str(crop_TMP_SIZE_FOR_MODEL) + "px."]
        save_string_to_file(strings, output_measurement_viz + '_settings.txt')

        print("################## Saving Last Experiment info ##################")
        print("bboxes_per_frames array", len(bboxes_per_frames[0]))
        print("crop_per_frames array", len(crop_per_frames[0]))
        print("crop_TMP_SIZE_FOR_MODEL", crop_TMP_SIZE_FOR_MODEL)

        dict = {}
        dict["bboxes_per_frames"] = bboxes_per_frames
        dict["crop_per_frames"] = crop_per_frames
        dict["crop_TMP_SIZE_FOR_MODEL"] = crop_TMP_SIZE_FOR_MODEL
        saveDict(dict,output_savedLastExp)
        #saveDict, loadDict

    else: # reuse_last_experiment is True
        print("#!!!!!!!!!!!!!!# WARNING, reusing last experiment #!!!!!!!!!!!!!!#")
        print("################## Loading Last Experiment info ##################")
        dict = loadDict(output_savedLastExp)
        bboxes_per_frames = dict["bboxes_per_frames"]
        crop_per_frames = dict["crop_per_frames"]
        crop_TMP_SIZE_FOR_MODEL = dict["crop_TMP_SIZE_FOR_MODEL"]
        print("bboxes_per_frames array", len(bboxes_per_frames[0]))
        print("crop_per_frames array", len(crop_per_frames[0]))
        print("crop_TMP_SIZE_FOR_MODEL", crop_TMP_SIZE_FOR_MODEL)


    print("################## Annotating frames ##################")
    iou_threshold = 0.5 # towards 0.01 its more drastic and deletes more bboxes which are overlapped
    limit_prob_lowest = 0 #0.70 # inside we limited for 0.3

    print_first = True
    annotations_names_saved = []
    annotations_lines_saved = []

    import tensorflow as tf
    sess = tf.Session()
    colors = annotate_prepare()

    for frame_i in range(0,len(frame_files)):
        test_bboxes = bboxes_per_frames[frame_i]
        from_number = len(test_bboxes)

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
            copyfile(INPUT_FRAMES + frame_files[frame_i], output_frames_folder + frame_files[frame_i])
            continue

        person_id = 0

        DEBUG_TURN_OFF_NMS = False
        if not DEBUG_TURN_OFF_NMS:
            """
            nms_arrays = py_cpu_nms(arrays, iou_threshold)
            reduced_bboxes_1 = []
            for j in range(0,len(nms_arrays)):
                a = ['person',nms_arrays[j],0.0,person_id]
                reduced_bboxes_1.append(a)
            """
            nms_arrays, scores = non_max_suppression_tf(sess, arrays,scores,allowed_number_of_boxes,iou_threshold)
            reduced_bboxes_2 = []
            for j in range(0,len(nms_arrays)):
                a = ['person',nms_arrays[j],scores[j],person_id]
                reduced_bboxes_2.append(a)

            test_bboxes = reduced_bboxes_2

        print("in frame", frame_i, "reduced from", from_number, "to", len(test_bboxes), "bounding boxes with NMS.")

        if SETTINGS["postprocess_merge_splitline_bboxes"]:
            replace_test_bboxes = postprocess_bboxes_by_splitlines(crop_per_frames[frame_i], test_bboxes, overlap_px_h=SETTINGS["overlap_px"], DEBUG_POSTPROCESS_COLOR=SETTINGS["debug_color_postprocessed_bboxes"])
            #test_bboxes += replace_test_bboxes
            test_bboxes = replace_test_bboxes

        if print_first:
            print("Annotating with bboxes of len: ", len(test_bboxes) ,"files in:", INPUT_FRAMES + frame_files[frame_i], ", out:", output_frames_folder + frame_files[frame_i])
            print_first = False

        img = annotate_image_with_bounding_boxes(INPUT_FRAMES + frame_files[frame_i], output_frames_folder + frame_files[frame_i], test_bboxes, colors,
                                           draw_text=False, save=True, show=False, thickness=SETTINGS["thickness"])
        img_size = img.size

        if SETTINGS["annotate_frames_with_gt"]:
            annotation_name = frame_files[frame_i][:-4]
            annotation_path = annotation_name + ".xml"
            if annotation_path in annotation_files:
                # we have ground truth for this file, we would like to save the predicted annotations
                # <image identifier> <confidence> <left> <top> <right> <bottom>

                for bbox in test_bboxes:
                    predicted_class = bbox[0]

                    if predicted_class is 'crop':
                        continue

                    box = bbox[1]
                    score = bbox[2]
                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(img_size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(img_size[0], np.floor(right + 0.5).astype('int32'))

                    line = str(annotation_name)+" "+str(score)+" "+str(left)+" "+str(top)+" "+str(right)+" "+str(bottom)

                    annotations_lines_saved.append(line)
                annotations_names_saved.append(str(annotation_name))

    if SETTINGS["annotate_frames_with_gt"]:
        print(len(annotations_lines_saved), annotations_lines_saved[0:3])

        with open(output_annotation+'names.txt', 'w') as the_file:
            for l in annotations_names_saved:
                the_file.write(l+'\n')
        with open(output_annotation+'bboxes.txt', 'w') as the_file:
            for l in annotations_lines_saved:
                the_file.write(l+'\n')

    sess.close()


    print("################## Cleanup ##################")

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
parser.add_argument('-horizontal_splits', help='number or horizontal splits in image', default='2')
parser.add_argument('-overlap_px', help='overlap in pixels', default='20')
parser.add_argument('-atthorizontal_splits', help='number or horizontal splits in image for attention model', default='1')
parser.add_argument('-input', help='path to folder full of frame images',
                    default="/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/")
parser.add_argument('-name', help='run name - will output in this dir', default='_Test-'+day+month)
parser.add_argument('-attention', help='use guidance of automatic attention model', default='True')
parser.add_argument('-thickness', help='thickness', default='10,2')
parser.add_argument('-extendmask', help='extend mask by', default='300')
parser.add_argument('-startframe', help='start from frame index', default='0')
parser.add_argument('-attframespread', help='look at attention map of this many nearby frames - minus and plus', default='0')
parser.add_argument('-annotategt', help='annotate frames with ground truth', default='False')
parser.add_argument('-reuse_last_experiment', help='Reuse last experiments bounding boxes? Skips the whole evaluation part.', default='False')
parser.add_argument('-postprocess_merge_splitline_bboxes', help='PostProcessing merging closeby bounding boxes found on the edges of crops.', default='True')

parser.add_argument('-debug_save_masks', help='DEBUG save masks? BW outlines of attention model. accepts "one" or "all"', default='one')
parser.add_argument('-debug_save_crops', help='DEBUG save crops? Attention models crops. accepts "one" or "all"', default='False')
parser.add_argument('-debug_color_postprocessed_bboxes', help='DEBUG color postprocessed bounding boxes?', default='False')

parser.add_argument('-anchorf', help='anchor file', default='yolo_anchors.txt')


if __name__ == '__main__':
    args = parser.parse_args()

    INPUT_FRAMES = args.input
    RUN_NAME = args.name
    SETTINGS = {}
    SETTINGS["attention_horizontal_splits"] = float(args.atthorizontal_splits)
    SETTINGS["overlap_px"] = int(args.overlap_px)
    SETTINGS["horizontal_splits"] = int(args.horizontal_splits)
    SETTINGS["anchorfile"] = args.anchorf
    SETTINGS["startframe"] = int(args.startframe)
    SETTINGS["attention"] = (args.attention == 'True')
    SETTINGS["annotate_frames_with_gt"] = (args.annotategt == 'True')
    SETTINGS["extend_mask_by"] = int(args.extendmask)
    SETTINGS["att_frame_spread"] = int(args.attframespread)
    thickness = str(args.thickness).split(",")
    SETTINGS["thickness"] = [float(thickness[0]), float(thickness[1])]
    SETTINGS["allowed_number_of_boxes"] = 500
    SETTINGS["reuse_last_experiment"] = (args.reuse_last_experiment == 'True')
    SETTINGS["postprocess_merge_splitline_bboxes"] = (args.postprocess_merge_splitline_bboxes == 'True')

    SETTINGS["debug_save_masks"] = args.debug_save_masks
    SETTINGS["debug_save_crops"] = (args.debug_save_crops == 'True')
    SETTINGS["debug_color_postprocessed_bboxes"] = (args.debug_color_postprocessed_bboxes == 'True')

    #INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/liverpool_station_8k/input/frames_24fps/"
    #INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/hall_london_8k/inputs/frames_24fps/"
    #INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/input/frames_4fps_frombridge/"
    #INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/input/frames_29b/"
    #SETTINGS["startframe"] = 1

    #SETTINGS["att_frame_spread"] = 4 # should be cca +-1 sec - so 30 in 30fps video
    #INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/input/annotated/"
    #INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames_all/"
    #INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/hand_annot/"
    #INPUT_FRAMES = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/input/auto_annot/"
    INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/test_nms/input/frames3/"

    RUN_NAME = "_testNMSworking_v3"
    #RUN_NAME = "_variableResolution_2att_6high_allow500bboxes"

    SETTINGS["attention"] = True
    #SETTINGS["attention"] = False
    #SETTINGS["annotate_frames_with_gt"] = True

    SETTINGS["attention_horizontal_splits"] = 2
    SETTINGS["horizontal_splits"] = 6
    #SETTINGS["overlap_px"] = 20

    SETTINGS["reuse_last_experiment"] = True

    print(RUN_NAME, SETTINGS)
    main_sketch_run(INPUT_FRAMES, RUN_NAME, SETTINGS)
