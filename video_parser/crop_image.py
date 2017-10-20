import math

width = 3840
height = 2160

crop_sizes_possible = [288,352,416,480,544]

crop = 288
overlap_percent = 0.50

block = crop * overlap_percent
to = int(math.ceil(width / block))

##


##

for i in range(0,int(to)-1):
    w_from = int(i * (width/float(to)))
    if i is to-2:
        w_from = width - crop

    if i > 0:
        print (w_to - w_from) / float(crop)

    w_to = w_from + crop
    print w_from, w_to


print width - w_to


