import os
import numpy, random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

folder_with_frames = "/home/ekmek/intership_project/video_parser/PL_Pizza/_videoframes/"
crops_folder = "/home/ekmek/intership_project/video_parser/PL_Pizza/"

# from folder_with_frames/image000.jpg -> to crops_folder/image000/crop_0_0.jpg ... crop_n_n.jpg

frame_files = sorted(os.listdir(folder_with_frames))

for i in range(0,len(frame_files)):
    frame_path = folder_with_frames+frame_files[i]

    #img = Image.open(frame_path)
    print frame_path