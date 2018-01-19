# call yolo code on our own data
# my data will come in list of images
# i want to get measurements of both time and accuracy while using yolo v2
from data_handler import use_path_which_exists, get_data_from_folder, get_data_from_list
from visualize_time_measurement import visualize_time_measurements
import numpy as np

#@profile
def run_yolo(num_crops_per_frames, crop_per_frames, fixbb_crop, INPUT_FRAMES, frame_files, resize_frames=None, show_viz = False,
             model_h5='yolo.h5', anchors_txt='yolo_anchors.txt', VERBOSE=1):

    yolo_paths = ["/home/ekmek/YAD2K/", "/home/vruzicka/storage_pylon2/YAD2K/"]

    path_to_yolo = use_path_which_exists(yolo_paths)

    print (path_to_yolo)

    import sys,site
    site.addsitedir(path_to_yolo)
    #print (sys.path)  # Just verify it is there
    import yad2k, eval_yolo, eval_yolo_direct_images

    ################################################################
    num_frames = len(num_crops_per_frames)
    image_names, ground_truths, frame_ids, crop_ids = get_data_from_list(crop_per_frames)
    print (len(image_names), image_names[0:2])

    args = {}

    #model_h5 = 'yolo_832x832.h5'
    args["anchors_path"]=path_to_yolo+'model_data/' + anchors_txt
    args["classes_path"]=path_to_yolo+'model_data/coco_classes.txt'
    args["model_path"]=path_to_yolo+'model_data/' + model_h5
    args["score_threshold"]=0.3
    args["iou_threshold"]=0.5
    args["output_path"]=''
    args["test_path"]=''
    print(args)

    #evaluation_times, additional_times, bboxes = eval_yolo._main(args, input_paths, ground_truths, output_paths, num_frames, num_crops_per_frames,
    #                                                             save_annotated_images=False, verbose=VERBOSE, person_only=True)


    full_path_frame_files = [INPUT_FRAMES + s for s in frame_files]
    pureEval_times, ioPlusEval_times, bboxes = eval_yolo_direct_images._main(args, frames_paths=full_path_frame_files, crops_bboxes=crop_per_frames, crop_value=fixbb_crop, resize_frames=resize_frames, verbose=VERBOSE, person_only=True)

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

        crops_in_frame = crop_per_frames[frame_index]
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
                max_limit = fixbb_crop

                #bbox_aray = np.maximum(bbox_aray,[0,0,0,0])
                #bbox_aray = np.minimum(bbox_aray,[max_limit,max_limit,max_limit,max_limit])
                fix_array = bbox_array + [a_top, a_left, a_top, a_left]

                bboxes_per_frames[frame_index].append([bbox[0],fix_array,bbox[2],bbox[3]])

            bboxes_per_frames[frame_index] += fixed_bboxes
        bboxes_per_frames[frame_index] += debug_bbox

    avg_evaltime = np.array(pureEval_times[1:]).mean()
    avg_addtime = np.array(ioPlusEval_times[1:]).mean()
    print("Mean times (ignoring first):", avg_evaltime, "eval, ", avg_addtime, "eval plus io")

    if show_viz:
        print ("--------------===========================--------------")
        print ("Evaluation:", pureEval_times)
        print ("Evaluation plus OS:", ioPlusEval_times)
        evaluation_times_ = pureEval_times[1:]
        additional_times_ = ioPlusEval_times[1:]
        visualize_time_measurements([evaluation_times_, additional_times_], ["Evaluation", "Additional"])

    return pureEval_times, ioPlusEval_times, bboxes_per_frames
