# call yolo code on our own data
# my data will come in list of images
# i want to get measurements of both time and accuracy while using yolo v2
from data_handler import use_path_which_exists
import argparse

yolo_paths = ["/home/ekmek/YAD2K/", "/home/vruzicka/storage_pylon2/YAD2K/"]

path_to_yolo = use_path_which_exists(yolo_paths)

print (path_to_yolo)

import sys,site
site.addsitedir(path_to_yolo)
print (sys.path)  # Just verify it is there
import yad2k, test_yolo


parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    '-model_path', help='path to h5 model file containing body of a YOLO_v2 model',
    default=path_to_yolo+'model_data/yolo.h5')
parser.add_argument(
    '--anchors_path',help='path to anchors file, defaults to yolo_anchors.txt',
    default=path_to_yolo+'model_data/yolo_anchors.txt')
parser.add_argument(
    '--classes_path', help='path to classes file, defaults to coco_classes.txt',
    default=path_to_yolo+'model_data/coco_classes.txt')
parser.add_argument(
    '--test_path', help='path to directory of test images, defaults to images/',
    default=path_to_yolo+'images')
parser.add_argument(
    '--output_path', help='path to output test images, defaults to images/out',
    default=path_to_yolo+"images/out")
parser.add_argument(
    '--score_threshold', type=float, help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '--iou_threshold', type=float, help='threshold for non max suppression IOU, default .5',
    default=.5)

evaluation_times = test_yolo._main(parser.parse_args())

print ("--------------===========================--------------")
print (evaluation_times)

