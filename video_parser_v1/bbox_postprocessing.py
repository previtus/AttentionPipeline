import numpy as np

def between(x,a,b):
    if x >= a and x <= b:
        return True
    return False

def join_splitlines(a,b):
    # splitline comes in [left, middle, right, middle]
    left = np.min([a[0],b[0]])
    right = np.max([a[2],b[2]])
    return [left, a[1], right, a[1]]

def splitlines_intersect_horizontally(a,b):
    # splitline comes in [left, middle, right, middle]
    if a[1] == b[1]:
        a_left = a[0]
        b_left = b[0]
        a_right = a[2]
        b_right = b[2]
        if between(a_right, b_left, b_right) or between(a_left, b_left, b_right):
            return True
    return False

def crops_intersect_vertically(a,b):
    # both a,b are crops with (_,[left,bottom,right,top])
    a_bottom = a[1][1]
    a_top = a[1][3]

    b_bottom = b[1][1]
    b_top = b[1][3]

    a_left = a[1][0]
    b_left = b[1][0]
    a_right = a[1][2]
    b_right = b[1][2]

    if a_top == b_top or b_bottom == a_bottom:
        return False

    if a_top > b_bottom and a_top < b_top:
        if a_left > b_left and a_left < b_right:
            return True
        elif a_right >= b_left and a_right <= b_right:
            return True
    elif a_bottom < b_top and a_bottom > b_bottom:
        if a_left >= b_left and a_left <= b_right:
            return True
        elif a_right >= b_left and a_right <= b_right:
            return True
    return False

def vertical_splitline_between(a,b):
    # both a,b are crops with (_,[left,bottom,right,top])
    a_bottom = a[1][1]
    a_top = a[1][3]

    b_bottom = b[1][1]
    b_top = b[1][3]

    top = np.max([a_top,a_bottom,b_top,b_bottom])
    bottom = np.min([a_top,a_bottom,b_top,b_bottom])

    middle = (top + bottom) / 2.0

    a_left = a[1][0]
    b_left = b[1][0]
    # we want the rightmost one
    left = np.max([a_left,b_left])

    a_right = a[1][2]
    b_right = b[1][2]
    # we want the leftmost one
    right = np.min([a_right,b_right])

    return [left, middle, right, middle]

def get_splitlines(cropboxes):
    # structure of bounding box = top, left, bottom, right, (top < bottom), (left < right)
    # structure of crop = left, bottom, right, top, (bottom < top),(left < bottom)

    N = len(cropboxes)
    ignore_matrix = np.ones((N, N), dtype=bool)
    for i in range(0,N):
        ignore_matrix[i,i] = False

    splitlines = []

    for i in range(0,N):
        for j in range(i,N):
            if ignore_matrix[i,j]:
                # we can now work with crop_i and crop_j
                cropi = cropboxes[i]
                cropj = cropboxes[j]
                if crops_intersect_vertically(cropi,cropj):
                    #print("split line between", i,j)
                    ignore_matrix[i,j] = False
                    ignore_matrix[j,i] = False
                    splitline = vertical_splitline_between(cropi,cropj)
                    splitlines.append(splitline)

    #print(ignore_matrix)

    #print("splitlines before", len(splitlines), splitlines)

    # test with randomly mixing the list = if its robust

    M = len(splitlines)
    ignore_vector = np.ones((M), dtype=bool)

    final_list = []

    for i in range(0,M):
        if ignore_vector[i]:
            splitline_a = splitlines[i]
            ignore_vector[i] = False

            for j in range(i,M):
                if ignore_vector[j]:
                    splitline_b = splitlines[j]

                    if splitlines_intersect_horizontally(splitline_a, splitline_b):
                        splitline_a = join_splitlines(splitline_a, splitline_b)
                        #print("joined", i, j)
                        ignore_vector[j] = False

            #print(i, "final", splitline_a)
            final_list.append(splitline_a)

    #print("pre splitlines", len(final_list), final_list)
    #final_list = set(tuple(i) for i in final_list)

    #print("final splitlines", len(splitlines), splitlines)
    splitlines = final_list
    return splitlines

def process_bboxes_near_splitlines(splitlines, bboxes, overlap_px_h, threshold_for_ratio, DEBUG_POSTPROCESS_COLOR):
    # for each suspicious line
    new_bounding_boxes = []
    indices_to_cancel = []
    debug = []

    for splitline in splitlines:
        #print("splitline", splitline)
        # find all bboxes above and bellow with distance < overlap
        split_height = splitline[1]

        bellow_bboxes = []
        allowed_bellow = []
        bellow_indices = []
        above_bboxes = []
        allowed_above = []
        above_indices = []

        for i, bbox in enumerate(bboxes):
            #print(bbox)
            # bounding box = top, left, bottom, right
            # distance between bbox <> splitline
            top = bbox[1][0]
            bottom = bbox[1][2]
            d1 = abs(top - split_height)
            d2 = abs(bottom - split_height)
            d = min(d1,d2)

            if d < overlap_px_h:
                # eliminate those with wrong aspect ratio
                w = abs(bbox[1][1] - bbox[1][3])
                h = abs(bbox[1][0] - bbox[1][2])

                #print("h,w,h/w",h,w,(h/w))

                half = (top + bottom) / 2.0
                side = split_height - half
                #print("side", side)

                # sign of side -> negative: bellow, positive: above
                condition = ((h / w) > threshold_for_ratio)

                if side < 0.0:
                    bellow_bboxes.append(bbox)
                    bellow_indices.append(i)
                    allowed_bellow.append(not condition)
                else:
                    above_bboxes.append(bbox)
                    above_indices.append(i)
                    allowed_above.append(not condition)

        close_bboxes = []
        # for each couple <above>, <bellow> we see if they are close and at least one is allowed
        for a, above in enumerate(above_bboxes):
            for b, bellow in enumerate(bellow_bboxes):
                if allowed_above[a] or allowed_bellow[b]:
                    # if these two are close enough - vertically + horizontally

                    # bounding box = top, left, bottom, right
                    h1 = abs(above[1][0] - bellow[1][2])  # top1 - bottom2
                    h2 = abs(above[1][2] - bellow[1][0])  # bottom1 - top2
                    h = min(h1, h2)

                    l = abs(above[1][1] - bellow[1][1])  # left1 - left2
                    r = abs(above[1][3] - bellow[1][3])  # right1 - right2

                    ### THRESHOLDs
                    # h < L*overlap_px_h
                    # l + r < K*overlap_px_h
                    H_multiple = 2.0
                    W_multiple = 4.0

                    if h < H_multiple * overlap_px_h:
                        if (l+r) < W_multiple * overlap_px_h:

                            # top, left, bottom, right (bottom < top),(left < bottom)
                            top = min(above[1][0], bellow[1][0])
                            left = min(above[1][1], bellow[1][1])
                            bottom = max(above[1][2], bellow[1][2])
                            right = max(above[1][3], bellow[1][3])
                            location_merge = [top,left,bottom,right]
                            #probability_merge = (above[2] + bellow[2]) / 2.0 # Average
                            probability_merge = max(above[2],bellow[2]) # Max
                            color = 0
                            if DEBUG_POSTPROCESS_COLOR:
                                color = 9
                            merged_bbox = ['person', location_merge, probability_merge, color]

                            close_bboxes.append(above)
                            close_bboxes.append(bellow)
                            new_bounding_boxes.append(merged_bbox)
                            i = above_indices[a]
                            j = bellow_indices[b]
                            indices_to_cancel.append(i)
                            indices_to_cancel.append(j)

        debug += close_bboxes

    keep_bboxes = []
    for i, bbox in enumerate(bboxes):
        if i not in indices_to_cancel:
            keep_bboxes.append(bbox)

    return new_bounding_boxes, keep_bboxes

def postprocess_bboxes_by_splitlines(cropboxes, bboxes, overlap_px_h, DEBUG_POSTPROCESS_COLOR, DEBUG_SHOW_LINES=False):
    # structure of bounding box = top, left, bottom, right, (top < bottom), (left < right)
    # structure of crop = left, bottom, right, top, (bottom < top),(left < bottom)
    threshold_for_ratio = 2.0

    splitlines = get_splitlines(cropboxes)

    new_bounding_boxes, keep_bboxes = process_bboxes_near_splitlines(splitlines, bboxes, overlap_px_h, threshold_for_ratio, DEBUG_POSTPROCESS_COLOR)

    debug_add = []
    if DEBUG_SHOW_LINES:
        for splitline in splitlines:

            left = splitline[0]
            bottom = splitline[1]
            right = splitline[2]
            top = splitline[3]

            debug_add.append(['person', [top, left, bottom, right], 1.0, 6])

    return new_bounding_boxes+keep_bboxes+debug_add
