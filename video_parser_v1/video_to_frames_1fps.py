# Idea of this project is - parsing high resolution videos of variable resolution, framerate, etc. and saving each frame into a folder

import os
import cv2
import argparse

def video2frames(pathIn, pathOut):
    # inputs
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # CV2 not installed
    #cap = cv2.VideoCapture("Exchanging_bags_day_indoor_1_original.mp4")
    #length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = cap.get(cv2.CAP_PROP_FPS)

    #print (length, "frames", "@", fps, width, "x", height)

    import subprocess
    # ffmpeg -i Exchanging_bags_day_indoor_1_original.mp4 -vf fps=1 test_1fps/frame_%d.png
    os.chdir(os.path.dirname(pathIn))
    command = ['ffmpeg', '-i', os.path.basename(pathIn), '-vf', 'fps=1', pathOut+"%04d.jpg"]
    print(command)
    subprocess.call(command)

    image_files = os.listdir(pathOut)
    return image_files

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathIn", default="/home/ekmek/intership_project/video_parser_v1/_videos_to_test/DrivingNY/Driving Downtown - 42nd St Theaters - New York City 4K.mp4")
    parser.add_argument("--pathOut", default="/home/ekmek/intership_project/video_parser_v1/_videos_to_test/DrivingNY/frames/")
    parser.add_argument("--toFrame", default="-1")
    args = parser.parse_args()
    #print(args)

    frames = video2frames(args.pathIn, args.pathOut) #args.toFrame
    print (frames)