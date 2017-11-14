voc2012 = "/home/ekmek/Downloads/datasets/pascal voc 2012/VOCdevkit/VOC2012/JPEGImages/"
cat2000 = "/home/ekmek/Downloads/datasets/CAT2000 saliency/trainSet/Stimuli/Action/"
salicon_challenge = "/home/ekmek/Downloads/datasets/salicon-2017-challenge/images/"
mit300 = "/home/ekmek/Downloads/datasets/MIT300/BenchmarkIMAGES/BenchmarkIMAGES/"
examples = "/home/ekmek/saliency_tools/_sample_inputs/images/"


folder = examples

import os
import numpy
from scipy import misc

from os import listdir
from os.path import isfile, join
image_files = [f for f in listdir(folder) if isfile(join(folder, f))]

hs = []
ws = []
ds = []

for i in range(0,min(100,len(image_files))):

    img_path = folder+image_files[i]
    #print img_path

    img = misc.imread(img_path)
    h = len(img)
    w = len(img[0])
    d = len(img[0][0])

    #print img_path, h, w, d, img.shape

    hs.append(h)
    ws.append(w)
    ds.append(d)

hs = numpy.asarray(hs)
ws = numpy.asarray(ws)
ds = numpy.asarray(ds)

print "heights", hs.mean(), "max", hs.max(), "min", hs.min()
print "widths", ws.mean(), "max", ws.max(), "min", ws.min()
print "depths", ds.mean(), "max", ds.max(), "min", ds.min()
