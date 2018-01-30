from voc_eval import voc_eval, parse_rec, voc_eval_twoArrs
from mark_frame_with_bbox import annotate_image_with_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np

gt_path_folder = '/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/input/annotated/'
output_model_predictions_folder = '/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/output_annotation_results/'

gt_path_folder = '/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/input/frames_all/'
output_model_predictions_folder = '/home/ekmek/intership_project/video_parser/_videos_to_test/PL_Pizza sample/output_annotation_results/'

# EXAMPLE
gt_path_folder = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/input/auto_annot/"
output_model_predictions_folder = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/output_annotation_results/"


imagesetfile = output_model_predictions_folder+"annotnames.txt"
predictions_file  = output_model_predictions_folder+"annotbboxes.txt"
rec, prec, ap = voc_eval(predictions_file,gt_path_folder,imagesetfile,'person')

print("ap", ap)

with open(predictions_file, 'r') as f:
    lines = f.readlines()
predictions = [x.strip().split(" ") for x in lines]

predictions_dict = {}

for pred in predictions:
    score = float(pred[1])
    # <image identifier> <confidence> <left> <top> <right> <bottom>
    left   = int(pred[2])
    top    = int(pred[3])
    right  = int(pred[4])
    bottom = int(pred[5])
    arr = [score, left, top, right, bottom]
    if not pred[0] in predictions_dict:
        predictions_dict[pred[0]] = []
    predictions_dict[pred[0]].append(arr)

print("predictions",len(predictions_dict), predictions_dict)

with open(imagesetfile, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]

aps = []

for imagename in imagenames:
    img = gt_path_folder + imagename + ".jpg"
    gt_file = gt_path_folder + imagename + ".xml"

    gt = parse_rec(gt_file)
    predictions = predictions_dict[imagename]

    print(imagename)
    print("ground truth:", len(gt), gt)
    print("predictions:", len(predictions), predictions)

    colors = [(0,128,0,125), (255, 165,0,125)] # green=GT orange=PRED

    bboxes_gt = []
    bboxes_pred = []
    c_gt = 0
    for i in gt:
        bb = i["bbox"]
        print(bb)
        # left, top, right, bottom => top, left, bottom, right
        bb = [bb[1], bb[0], bb[3], bb[2]]
        bboxes_gt.append(['person_gt', bb, 1.0, c_gt])

    print("-")


    c_pred = 1
    for p in predictions:
        print(p)
        bb = p[1:]
        bb = [bb[1], bb[0], bb[3], bb[2]]
        bboxes_pred.append(['person', bb, p[0], c_pred])

    draw_text = True

    bboxes = bboxes_gt + bboxes_pred
    #bboxes = bboxes_gt
    img = annotate_image_with_bounding_boxes(img, "", bboxes, colors, ignore_crops_drawing=True, draw_text=draw_text,
                                       show=False, save=False, thickness=[4.0, 1.0], resize_output = 1.0)

    rec, prec, ap = voc_eval_twoArrs(bboxes_gt,bboxes_pred,ovthresh=0.5)

    fig = plt.figure()
    plt.imshow(img)
    plt.title("Frame "+imagename+", ap: "+str(ap))
    aps.append(ap)

    plt.show()
    plt.clf()

    print("")

print(aps)
print("[AP] min, max, avg:",np.min(aps), np.max(aps), np.mean(aps))
fig = plt.figure()
plt.title("AP over frames, avg: "+str(np.mean(aps)))
plt.plot(aps)
plt.show()
