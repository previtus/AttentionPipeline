import math

def get_crops_parameters(w, crop=288, over=0.5, scale=1.0):
    # OLD VERSION
    crop = scale * crop

    block = crop * (1.0 - over)
    pocet = (w - (crop - block)) / block
    pocet = max([pocet,1.0])
    nastejne = (w - (crop - block)) / int(pocet)

    offset = w - (int((int(pocet) - 1) * nastejne) + crop)
    balance = offset / 2.0

    params = []
    for i in range(0, int(pocet)):
        w_from = int(i * nastejne + balance)
        w_to = int(w_from + crop)
        params.append((w_from, w_to))

    #print (w - w_to)
    return params


def best_squares_overlap(w, h, horizontal_splits, overlap_px):

    crop = int((h + overlap_px * (horizontal_splits - 1)) / horizontal_splits)

    row_list = []
    for i in range(0, horizontal_splits):
        row_list.append([int(i * (crop - overlap_px)), int(i * (crop - overlap_px) + crop)])

    n_v = math.ceil((w - crop) / (crop - overlap_px) + 1)
    loc = (w - crop) / (n_v - 1)


    column_list = []

    column_list.append([0,crop])
    for i in range(1, n_v - 1):
        column_list.append([int(i*(loc)), int(i*(loc)+crop)])
    column_list.append([w-crop, w])

    #print(len(column_list) * len(row_list))

    return column_list, row_list
