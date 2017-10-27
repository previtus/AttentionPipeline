# call yolo code on our own data
# my data will come in list of images
# i want to get measurements of both time and accuracy while using yolo v2
from data_handler import use_path_which_exists, get_data
from visualize_time_measurement import visualize_time_measurements
import argparse
import numpy as np

def run_yolo(frames_folder, output_folder, fixbb_crop_per_frames, fixbb_scale, fixbb_crop, show_viz = False, ground_truth_file = None):

    yolo_paths = ["/home/ekmek/YAD2K/", "/home/vruzicka/storage_pylon2/YAD2K/"]

    path_to_yolo = use_path_which_exists(yolo_paths)

    print (path_to_yolo)

    import sys,site
    site.addsitedir(path_to_yolo)
    #print (sys.path)  # Just verify it is there
    import yad2k, eval_yolo

    parser = argparse.ArgumentParser(
        description='Run a YOLO_v2 style detection model on test images..')
    parser.add_argument(
        '-model_path', help='path to h5 model file containing body of a YOLO_v2 model',
        default=path_to_yolo+'model_data/yolo.h5')
    parser.add_argument(
        '--anchors_path',help='path to anchors file, defaults to yolo_anchors.txt',
        default=path_to_yolo+'model_data/yolo_anchors.txt')
    parser.add_argument(
        '--classes_path', help='path to classes file, defaults to coco_classes.txt',
        default=path_to_yolo+'model_data/coco_classes.txt')
    parser.add_argument(
        '--test_path', help='path to directory of test images, defaults to images/',
        default=path_to_yolo+'images')
    parser.add_argument(
        '--output_path', help='path to output test images, defaults to images/out',
        default=path_to_yolo+"images/out")
    parser.add_argument(
        '--score_threshold', type=float, help='threshold for bounding box scores, default .3',
        default=.3)
    parser.add_argument(
        '--iou_threshold', type=float, help='threshold for non max suppression IOU, default .5',
        default=.5)

    ################################################################
    image_names, ground_truths, frame_ids, crop_ids, num_frames, num_crops = get_data(frames_folder, ground_truth_file, dataset = 'ParkingLot')

    image_names = [val for sublist in image_names for val in sublist]
    frame_ids = [val for sublist in frame_ids for val in sublist]
    crop_ids = [val for sublist in crop_ids for val in sublist]

    #image_names = np.array(image_names).flatten()
    #frame_ids = np.array(frame_ids).flatten()
    #crop_ids = np.array(crop_ids).flatten()

    output_paths = [output_folder + s for s in image_names]
    input_paths = [frames_folder + s for s in image_names]

    #print (len(image_names), image_names[0:2])
    #print (len(input_paths), input_paths[0:2])
    #print (len(output_paths), output_paths[0:2])

    ## TESTS
    #limit = 60
    #image_names = image_names[0:limit]
    #input_paths = input_paths[0:limit]
    #output_paths = output_paths[0:limit]

    evaluation_times, additional_times, bboxes = eval_yolo._main(parser.parse_args(), input_paths, ground_truths, output_paths,
                                                                 save_annotated_images=False, verbose=1, person_only=False)
    bboxes_per_frames = []
    for i in range(0,num_frames):
        bboxes_per_frames.append([])

    for index in range(0,len(image_names)):
        frame_index = frame_ids[index] - frame_ids[0]
        crop_index = crop_ids[index]

        #if len(bboxes_per_frames) < frame_index+1:
        #    bboxes_per_frames.append([])
        if bboxes_per_frames[frame_index] is None:
            bboxes_per_frames[frame_index] = []

        crops_in_frame = fixbb_crop_per_frames[frame_index]
        current_crop = crops_in_frame[crop_index]

        #print("current_crop_coord", current_crop[1], fixbb_scale )
        #print("these bboxes need fixing:",len(bboxes[index]), bboxes[index])

        #debug_bbox = [['crop',[100,200,400,205],1.0,13]]
        a_left = current_crop[1][0]
        a_top = current_crop[1][1]
        a_right = current_crop[1][2]
        a_bottom = current_crop[1][3]
        debug_bbox = [['crop',[a_top,a_left,a_bottom,a_right],1.0,70]]

        if len(bboxes[index]) > 0: #not empty
            fixed_bboxes = []
            for bbox in bboxes[index]:
                bbox_array = bbox[1]
                max_limit = fixbb_crop * fixbb_scale

                #bbox_aray = np.maximum(bbox_aray,[0,0,0,0])
                #bbox_aray = np.minimum(bbox_aray,[max_limit,max_limit,max_limit,max_limit])
                fix_array = bbox_array * fixbb_scale + [a_top, a_left, a_top, a_left]

                bboxes_per_frames[frame_index].append([bbox[0],fix_array,bbox[2],bbox[3]])

            bboxes_per_frames[frame_index] += fixed_bboxes
        bboxes_per_frames[frame_index] += debug_bbox

    avg_evaltime = np.array(evaluation_times[1:]).mean()
    avg_addtime = np.array(additional_times[1:]).mean()
    print("Mean times (ignoring first):", avg_evaltime, "eval, ", avg_addtime, "additional")

    if show_viz:
        print ("--------------===========================--------------")
        print ("Evaluation:", evaluation_times)
        print ("Additional OS:", additional_times)
        evaluation_times = evaluation_times[1:]
        additional_times = additional_times[1:]
        visualize_time_measurements([evaluation_times, additional_times], ["Evaluation", "Additional"])

    return evaluation_times, bboxes_per_frames
"""
    #frames_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/set1_544_0.6/"
    #output_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set1_544_0.6/"
    
    #frames_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/set2_288_0.6/"
    #output_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set2_288_0.6/"
    
    frames_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/set3_1024_0.6/"
    output_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set3_1024_0.6/"
    
    #frames_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/set4_1472_0.6/"
    #output_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/OUT_set4_1472_0.6/"
    
    ground_truth_file = "/home/ekmek/intership_project/video_parser/PL_Pizza/PL_Pizza_GT.txt"


"""