# intership_project
internship project at CMU, 2017-2018

![Illustration image](https://github.com/previtus/intership_project/blob/master/video_parser/project_illustration.jpg)

Working with high resolution videos to locate certain objects. First pass serves as attention selection on low res version / fast check, second pass then checks more thoroughly on selected areas.

# Instructions

## Installation
With
- python 3.6.1, tensorflow with gpu and cuda support, keras
- YAD2K, python YOLO v2 implementation: https://github.com/allanzelener/YAD2K (commit hash a42c760ef868bc115e596b56863dc25624d2e756)
- put files from "__to-be-put-with-YAD2K" to YAD2K folder
- make sure that there is correct path to the YAD2K folder in "yolo_handler.py" on line `yolo_paths = ["/home/<whatever>/YAD2K/","<more possible paths>"]`
- prepare data *(see the ffmpeg commands bellow)* so it follows this hierarchy:
  * VideoName (whatever name, for example `PL_Pizza sample`)
    * input
      * frames (whatever name again, for example separate different fps)
        * 0001.jpg
        * 0002.jpg
        * ...

## Data preparation

Works with high resolution videos, respectively with the individual frames saved into input folder.
We can convert the resulting annotated output frames back into video.

**[Video to frames]** 30 images every second (30 fps, can be changed), named frames/0001.jpg, frames/0002.jpg, ...
- `ffmpeg -i VIDEO.mp4 -vf fps=30 frames/%04d.jpg`

**[Frames to video]** Keep the same framerate
- `ffmpeg -r 30/1 -pattern_type glob -i 'frames/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p out_30fps.mp4`

## Running

- Go through proper installation of everything required.
- `cd /<whatever>/intership_project/video_parser`
- `python run_fast_sketch.py -horizontal_splits 2 -attention_horizontal_splits 1 -input "/<path>/PL_Pizza sample/input/frames/" -name "_ExampleRunNameHere"`
- See the results in `/<path>/PL_Pizza sample/output_ExampleRunNameHere`

## Annotation
When the python code is run with `-annotategt 'True'`, then the model will look for which frames have ground truth annotations accompanying them (in VOC style .xml file next to the .jpg). For these frames it then saves results into the output folder (into files `annotbboxes.txt` and `annotnames.txt`).

Visualization tool can be then run (with paths to the input and output folder set correctly). For example set file "visualize_gt_prediction.py" with these paths:

`gt_path_folder = "/<path>/intership_project/_side_projects/annotation_conversion/annotated examples/input/auto_annot/"
output_model_predictions_folder = "/<path>/intership_project/_side_projects/annotation_conversion/annotated examples/output_annotation_results/"`

As result we should see something like image in  *_side_projects/annotation_conversion/annotated examples/example_visualization_of_ap_measurement.jpg*

- hand annotation possible with labelImg: https://github.com/tzutalin/labelImg



