# input frames images
# output marked frames images

def main_sketch_run(INPUT_FRAMES, SETTINGS):

    import os
    from crop_functions import crop_from_one_frame
    from yolo_handler import run_yolo
    from pathlib import Path

    video_file_root_folder = str(Path(INPUT_FRAMES).parents[1])
    crops_folder = video_file_root_folder + "/temporary/crops/"
    output_frames_folder = video_file_root_folder + "/output/frames/"
    if not os.path.exists(crops_folder):
        os.makedirs(crops_folder)
    if not os.path.exists(output_frames_folder):
        os.makedirs(output_frames_folder)

    # Frames to crops
    crop_per_frames = []
    frame_files = sorted(os.listdir(INPUT_FRAMES))
    for i in range(0, len(frame_files)):
        frame_path = INPUT_FRAMES + frame_files[i]

        crops = crop_from_one_frame(frame_path, crops_folder, SETTINGS["crop"], SETTINGS["over"], SETTINGS["scale"], show=False, save=True)
        crop_per_frames.append(crops)

    # Run YOLO on crops
    tmp = video_file_root_folder + "/temporary/tmp/"

    evaluation_times, bboxes = run_yolo(crops_folder, tmp)

    print (evaluation_times[0:5])

    print (len(bboxes))
    for bbox in bboxes:
        if len(bbox)>0:
            print (bbox)


INPUT_FRAMES = "/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames/"
SETTINGS = {}
SETTINGS["crop"] = 544 ## crop_sizes_possible = [288,352,416,480,544] # multiples of 32
SETTINGS["over"] = 0.6
SETTINGS["scale"] = 2.0


main_sketch_run(INPUT_FRAMES, SETTINGS)