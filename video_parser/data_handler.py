import os

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

def get_data(frames_folder, ground_truth_file, dataset):
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
    num_crops = len(crops)

    return image_paths, ground_truths, frame_ids, crop_ids, num_frames, num_crops

