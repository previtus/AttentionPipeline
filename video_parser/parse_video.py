# Idea of this project is - parsing high resolution videos of variable resolution, framerate, etc. and saving each frame into a folder

import os
import cv2
import argparse

def extractImages(pathIn, pathOut, toFrame=-1):
    # inputs
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    toFrame = int(toFrame)

    # convert
    import skvideo.io
    cap = skvideo.io.VideoCapture("Exchanging_bags_day_indoor_1_original.mp4")

    count = 0
    success = True
    while success:
        if toFrame is not -1 and count > toFrame:
            break
        success, image = cap.read()
        print ('Read frame (',count,'): ', success)
        if success:
            cv2.imwrite( pathOut + "frame%d.jpg" % count, image, [cv2.IMWRITE_JPEG_QUALITY, 50])     # save frame as JPEG file
            count += 1


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathIn", default="/home/ekmek/video_parser/big_buck_bunny_720p_5mb.mp4")
    parser.add_argument("--pathOut", default="test/")
    parser.add_argument("--toFrame", default="-1")
    args = parser.parse_args()
    #print(args)

    extractImages(args.pathIn, args.pathOut, args.toFrame)