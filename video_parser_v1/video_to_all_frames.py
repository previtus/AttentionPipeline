# Idea of this project is - parsing high resolution videos of variable resolution, framerate, etc. and saving each frame into a folder

import os
import cv2
import argparse

def video2frames(pathIn, pathOut, toFrame=-1):
    # inputs
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    toFrame = int(toFrame)

    """

    # convert
    import skvideo.io
    cap = skvideo.io.VideoCapture("Exchanging_bags_day_indoor_1_original.mp4")
    metadata = cap.get_info()
    print (metadata)

    video_stream = metadata["streams"][0]
    length = int(video_stream["nb_frames"])
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    framerate_string = video_stream["avg_frame_rate"].split("/")
    fps = int( float(framerate_string[0]) / float(framerate_string[1]) )


    """
    # CV2 not installed
    cap = cv2.VideoCapture("Exchanging_bags_day_indoor_1_original.mp4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print (length, "frames", "@", fps, width, "x", height, (length/fps))

    desired_fps = 1.0

    import json
    #print(metadata.keys())
    #print(json.dumps(metadata["streams"], indent=4))
    #print(json.dumps(metadata["format"], indent=4))

    count = 0
    success = True
    frames = []
    while success:
        if toFrame is not -1 and count > toFrame:
            break
        success, image = cap.read()
        print ('Read frame (',count,'): ', success)
        if success:
            cv2.imwrite( pathOut + "frame%d.jpg" % count, image, [cv2.IMWRITE_JPEG_QUALITY, 50])     # save frame as JPEG file
            frames.append( pathOut + "frame%d.jpg" % count )
            count += 1
    return frames


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathIn", default="/home/ekmek/video_parser_v1/big_buck_bunny_720p_5mb.mp4")
    parser.add_argument("--pathOut", default="test/")
    parser.add_argument("--toFrame", default="-1")
    args = parser.parse_args()
    #print(args)

    frames = video2frames(args.pathIn, args.pathOut, 5) #args.toFrame
    print (frames)