# input frames images
# output marked frames images

def main_sketch_run(INPUT_FRAMES, SETTINGS):

    import os
    from crop_functions import crop_from_one_frame
    from yolo_handler import run_yolo
    from mark_frame_with_bbox import annotate_image_with_bounding_boxes
    from pathlib import Path

    video_file_root_folder = str(Path(INPUT_FRAMES).parents[1])
    crops_folder = video_file_root_folder + "/temporary/crops/"
    output_frames_folder = video_file_root_folder + "/output/frames/"
    if not os.path.exists(crops_folder):
        os.makedirs(crops_folder)
    if not os.path.exists(output_frames_folder):
        os.makedirs(output_frames_folder)

    # Frames to crops
    print("################## Cropping frames ##################")
    crop_per_frames = []
    frame_files = sorted(os.listdir(INPUT_FRAMES))
    print("##",len(frame_files),"of frames")

    for i in range(0, len(frame_files)):
        frame_path = INPUT_FRAMES + frame_files[i]

        crops = crop_from_one_frame(frame_path, crops_folder, SETTINGS["crop"], SETTINGS["over"], SETTINGS["scale"], show=False, save=True)
        crop_per_frames.append(crops)

    # Run YOLO on crops
    print("")
    print("################## Running Model ##################")

    tmp = video_file_root_folder + "/temporary/tmp/"

    evaluation_times, bboxes_per_frames = run_yolo(crops_folder, tmp, crop_per_frames, SETTINGS["scale"], SETTINGS["crop"])

    #bboxes_per_frames = sort_out_crop_coords_and_bboxes(crop_per_frames, bboxes)

    #print (len(bboxes_per_frames), bboxes_per_frames)
    #print (len(bboxes_per_frames[0]), bboxes_per_frames[0])
    #print (len(bboxes_per_frames[0][0]), bboxes_per_frames[0][0])

    print("################## Annotating frames ##################")

    for i in range(0,len(frame_files)):
        annotate_image_with_bounding_boxes(INPUT_FRAMES + frame_files[i], output_frames_folder + frame_files[i], bboxes_per_frames[i],
                                           draw_text=False, save=True, show=False)


    #print (evaluation_times[0:5])

"""
INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/"
SETTINGS = {}
SETTINGS["crop"] = 544 ## crop_sizes_possible = [288,352,416,480,544] # multiples of 32
SETTINGS["over"] = 0.6
SETTINGS["scale"] = 1.0


main_sketch_run(INPUT_FRAMES, SETTINGS)

"""
INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/bag exchange/input/frames/"
SETTINGS = {}
SETTINGS["crop"] = 1024 ## crop_sizes_possible = [288,352,416,480,544] # multiples of 32
SETTINGS["over"] = 0.6
SETTINGS["scale"] = 1.0


main_sketch_run(INPUT_FRAMES, SETTINGS)
