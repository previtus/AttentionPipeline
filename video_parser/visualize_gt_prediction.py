from voc_eval import voc_eval, parse_rec, voc_eval_twoArrs
from mark_frame_with_bbox import annotate_image_with_bounding_boxes
import matplotlib.pyplot as plt

gt_path_folder = '/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/input/annotated/'

imagesetfile = "/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/output_tmptests/annotnames.txt"
predictions_file  = "/home/ekmek/intership_project/video_parser/_videos_to_test/PittsMine/output_tmptests/annotbboxes.txt"
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

for imagename in imagenames:
    img = gt_path_folder + imagename + ".jpg"
    gt_file = gt_path_folder + imagename + ".xml"

    gt = parse_rec(gt_file)
    predictions = predictions_dict[imagename]

    print(imagename)
    print("ground truth:", len(gt), gt)
    print("predictions:", len(predictions), predictions)

    colors = ['green', 'orange']

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

    bboxes = bboxes_gt + bboxes_pred
    img = annotate_image_with_bounding_boxes(img, "", bboxes, colors, ignore_crops_drawing=True, draw_text=True,
                                       show=False, save=False, thickness=[4.0, 1.0], resize_output = 1.0)

    rec, prec, ap = voc_eval_twoArrs(bboxes_gt,bboxes_pred)

    #fig = plt.figure()
    plt.imshow(img)
    plt.title("Frame "+imagename+", ap: "+str(ap))
    plt.show()
    plt.clf()

    print("")
