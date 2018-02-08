import os
import numpy as np

def saveDict(dict, filename):
    to_be_saved = data = {'S': dict}
    np.save(open(filename, 'wb'), to_be_saved)

def loadDict(filename):
    loaded = np.load(open(filename, 'rb'))
    return loaded[()]['S']

def save_string_to_file(strings, file):
    text_file = open(file, "w")
    for string in strings:
        text_file.write(string)
        text_file.write("\n")
    text_file.close()

def use_path_which_exists(list_of_possible_paths):
    '''
    From a list of possible paths choose the one which exists.
    :param list_of_possible_paths: possible paths
    :return: working path
    '''
    used_path = ''
    assigned = False

    for path in list_of_possible_paths:
        if os.path.exists(path):
            used_path = path
            assigned = True

    if not assigned:
        print ("Error, cannot locate the path of project, will likely fail!")

    return used_path

def load_ground_truth_ParkingLot(ground_truth_file):
    return None

def get_data_from_folder(frames_folder, ground_truth_file, dataset):
    # load labels
    if dataset is 'ParkingLot':
        ground_truths = load_ground_truth_ParkingLot(ground_truth_file)
    else:
        ground_truths = None

    # load image paths
    frame_dirs = [x[0] for x in os.walk(frames_folder)][1:]
    frame_dirs.sort()

    image_paths = []
    frame_ids = []
    crop_ids = []

    num_frames = 0
    for frame_dir in frame_dirs:
        frame_num = frame_dir[-4:]

        crops = sorted(os.listdir(frame_dir))

        crop_ids.append(list(range(0,len(crops))))
        crops = [frame_num + "/" + s for s in crops]

        frame_ids.append( [int(num_frames) for s in crops] )
        image_paths.append(crops)
        num_frames += 1

    # unflattened
    return image_paths, ground_truths, frame_ids, crop_ids

def get_data_from_list(crop_per_frames):
    image_paths = []
    frame_ids = []
    crop_ids = []

    for frame_i in range(0,len(crop_per_frames)):
        crops_of_frame = crop_per_frames[frame_i]

        for crop_i in range(0,len(crops_of_frame)):
            crop = crops_of_frame[crop_i]

            full_path = crop[0]
            image_paths.append(full_path)
            frame_ids.append(frame_i)
            crop_ids.append(crop_i)

            #print(frame_i, crop_i, crop)

    # flattened!!!
    ground_truths = None
    return image_paths, ground_truths, frame_ids, crop_ids

import os
def is_non_zero_file(fpath):
    # thanks https://stackoverflow.com/a/15924160
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0