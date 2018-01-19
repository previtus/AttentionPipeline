# intership_project
internship project at CMU, 2017-2018

Working with high resolution videos to locate certain objects. First pass serves as attention selection on low res version / fast check, second pass then checks more thorougly on selected areas.

# instructions

## installation
With
- python 3.6.1, tensorflow with gpu and cuda support, keras
- YAD2K, python YOLO v2 implementation: https://github.com/allanzelener/YAD2K

## data preparation

Works with high resolution videos, respectively with the individual frames saved into input folder.
We can convert the resulting annotated output frames back into video.

**[Video to frames]** 30 images every second (30 fps, can be changed), named frames/0001.jpg, frames/0002.jpg, ...
- `ffmpeg -i VIDEO.mp4 -vf fps=30 frames/%04d.jpg`

**[Frames to video]** Keep the same framerate
- `ffmpeg -r 30/1 -pattern_type glob -i 'frames/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p out_30fps.mp4`

## running

- Go through proper installation of everything required.
- `cd /<whatever>/intership_project/video_parser`
- `python run_fast_sketch.py -horizontal_splits 2 -attention_horizontal_splits 1 -input "/<path>/_videos_files/PL_Pizza sample/input/frames/" -name "_ExampleRunNameHere"`
- See the results in `/<path>/_videos_files/PL_Pizza sample/output_ExampleRunNameHere`

## annotation
- hand annotation possible with labelImg: https://github.com/tzutalin/labelImg


