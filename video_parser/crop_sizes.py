import math



w = 3840
scale = 1.0

for crop in [288]: #[288,352,416,480,544]
    for over in [0.5]:
        #crop = 362
        #over = 0.6  # overlapping percent
        crop = scale * crop

        block = crop * (1.0 - over)
        pocet = (w - (crop - block)) / block
        nastejne = (w - (crop - block)) / int(pocet)
        #print pocet, nastejne

        for i in range(0, int(pocet)):
            w_from = int(i*nastejne)
            w_to = int(w_from + crop)
            print w_from, w_to

        print w - (i*nastejne + crop)



w = 3840
scale = 1.0

crop = 288
over = 0.5 # overlapping percent
crop = scale * crop

block = crop * (1.0 - over)
pocet = (w - (crop - block)) / block
nastejne = (w - (crop - block)) / int(pocet)

offset = w - (int((int(pocet)-1)*nastejne) + crop)
balance = offset / 2.0

for i in range(0, int(pocet)):
    w_from = int(i*nastejne + balance)
    w_to = int(w_from + crop)
    print w_from, w_to

print w - w_to
"""
width = 3840
height = 2160

crop_sizes_possible = [288,352,416,480,544]

crop = 288.0
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


"""