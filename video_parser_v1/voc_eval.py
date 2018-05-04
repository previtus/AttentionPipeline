# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import _pickle as cPickle
import numpy as np

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval_twoArrs(groundtruth,predictions,ovthresh=0.5,use_07_metric=False):
    # both groundtruth,predictions hold bounding boxes in:
    # ['person', bbox, score, color]

    confidence = np.array([x[2] for x in predictions])
    BB = np.array([x[1] for x in predictions])
    gtBB = np.array([x[1] for x in groundtruth])

    image_ids = np.array(['0001'] * len(BB)) # lets say its in one image

    class_recs = {}
    class_recs['0001'] = {}
    class_recs['0001']['bbox'] = np.array(gtBB)
    class_recs['0001']['difficult'] = np.array([False] * len(gtBB))
    class_recs['0001']['det'] = np.array([False] * len(gtBB))

    #print("BB", len(BB), BB)
    #print("image_ids", len(image_ids), image_ids)
    #print("confidence", len(confidence), confidence)
    #print("class_recs", len(class_recs), class_recs)


    ## BB is array of all bboxes
    ## image_ids tells to which frame it belongs
    ## confidence are scores
    ## class_recs holds ground truths - its a dict with dict[index frame name] -> array with bbox attrib

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        #print("BB",BB)
        #print("BBGT",BBGT)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            #print("max intersection ",np.max(inters))
            #print("inters",inters)
            #print("uni",uni)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    #print("fp, tp: ",fp, tp)

    npos = len(gtBB) # hack, none are marked as difficult

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
    #    os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for i, imagename in enumerate(imagenames):
        path = annopath+imagename+".xml"
        #print(path)

        recs[imagename] = parse_rec(path)
        if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))

    '''
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print ('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)
    '''

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    ## BB is array of all bboxes
    ## image_ids tells to which frame it belongs
    ## confidence are scores
    ## class_recs holds ground truths - its a dict with dict[index frame name] -> array with bbox attrib

    #print("BB", len(BB), BB)
    #print("image_ids", len(image_ids), image_ids)
    #print("confidence", len(confidence), confidence)
    #print("class_recs", len(class_recs), class_recs)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    #print("NPOSSSS ", npos)

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_eval__returnNumberOfGTObjects(detpath,
             annopath,
             imagesetfile,
             classname):

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for i, imagename in enumerate(imagenames):
        path = annopath+imagename+".xml"
        #print(path)

        recs[imagename] = parse_rec(path)
        if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))

    # extract gt objects for this class
    class_recs = {}
    npos = 0

    numbers_of_gt_bboxes_in_files = []
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        #print("R",R)
        #print("bbox",bbox)
        num_of_boxes = len(bbox)
        numbers_of_gt_bboxes_in_files.append(num_of_boxes)
    print("numbers_of_gt_bboxes_in_files",numbers_of_gt_bboxes_in_files)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    frame_number_to_bboxes_number = {}
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]

    ## BB is array of all bboxes
    ## image_ids tells to which frame it belongs
    for d,id in enumerate(imagenames):
        if id not in frame_number_to_bboxes_number:
            frame_number_to_bboxes_number[id] = [numbers_of_gt_bboxes_in_files[d],0]

    #print("frame_number_to_bboxes_number pred",frame_number_to_bboxes_number)

    #print("image_ids",image_ids)
    #print("imagenames",imagenames)
    for id in image_ids:
        frame_number_to_bboxes_number[id][1] = frame_number_to_bboxes_number[id][1] + 1
    #print("frame_number_to_bboxes_number after", frame_number_to_bboxes_number)
    return frame_number_to_bboxes_number

"""
gt = '/home/ekmek/intership_project/video_parser_v1/_videos_to_test/PittsMine/input/annotated/0013.xml'
obj = parse_rec(gt)
print(len(obj), obj)


gt_path = '/home/ekmek/intership_project/video_parser_v1/_videos_to_test/PittsMine/input/annotated/'

imagesetfile = "/home/ekmek/intership_project/video_parser_v1/_videos_to_test/PittsMine/output_tmptests/annotnames.txt"
predictions  = "/home/ekmek/intership_project/video_parser_v1/_videos_to_test/PittsMine/output_tmptests/annotbboxes.txt"
rec, prec, ap = voc_eval(predictions,gt_path,imagesetfile,'person')

print("rec", rec)
print("prec", prec)
print("ap", ap)
"""

"""
gt = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/input/test_s21_input/0018.xml"
obj = parse_rec(gt)
print(len(obj), obj)


gt_path = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/input/test_s21_input/"

imagesetfile = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/output_test_s21/annotnames.txt"
predictions  = "/home/ekmek/intership_project/_side_projects/annotation_conversion/annotated examples/output_test_s21/annotbboxes.txt"
rec, prec, ap = voc_eval(predictions,gt_path,imagesetfile,'person')

print("rec", rec)
print("prec", prec)
print("ap", ap)
"""

# format of bboxes?

"""
ixmin = np.maximum(BBGT[:, 0], bb[0])
iymin = np.maximum(BBGT[:, 1], bb[1])
ixmax = np.minimum(BBGT[:, 2], bb[2])
iymax = np.minimum(BBGT[:, 3], bb[3])
"""
# xmin maybe left = bb[0] == bbgt[*,0]
# ymin maybe bottom = bb[1] == bbgt[*,1]
# xmax maybe right = bb[2] == bbgt[*,2]
# ymax maybe top = bb[3] == bbgt[*,3]
## but top and bottom might be swapped

# [left, bottom, right, top]
# or
# [left, top, right, bottom]
# well anyway, should be
# [xmin, ymin, xmax, ymax] - easy