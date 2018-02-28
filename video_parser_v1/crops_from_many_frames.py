import os
from crop_functions import crop_from_one_frame

## crop_sizes_possible = [288,352,416,480,544] # multiples of 32
#crop = 544
#over = 0.6
#crops_folder = "/home/ekmek/intership_project/video_parser_v1/PL_Pizza/set1_544_0.6/"

crop = 1472
over = 0.6
crops_folder = "/home/ekmek/intership_project/video_parser_v1/PL_Pizza/set4_1472_0.6/"

folder_with_frames = "/home/ekmek/intership_project/video_parser_v1/PL_Pizza/_videoframes/"
ground_truth = "/home/ekmek/intership_project/video_parser_v1/PL_Pizza/PL_Pizza_GT.txt"

# from folder_with_frames/image000.jpg -> to crops_folder/image000/crop_0_0.jpg ... crop_n_n.jpg

frame_files = sorted(os.listdir(folder_with_frames))

for i in range(0,len(frame_files)):
#for i in range(0,2):
    frame_path = folder_with_frames+frame_files[i]

    #img = Image.open(frame_path)
    print (frame_path)

    scale = 1.0
    show = False
    save = True

    crop_from_one_frame(frame_path, crops_folder, crop, over, scale, show, save)

