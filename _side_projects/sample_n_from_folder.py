import os, fnmatch
from pathlib import Path
from shutil import copyfile


def sample_N_from_folder(path, N, target_folder):
    '''
    Copy Nth of the folder in path to target_folder
    :param path: path full of (lets say) images
    :param N: Nth
    :param target_folder: where to copy to, will clone the folder name
    :return:
    '''
    if path[-1] is not "/":
        path += "/"
    folder_name = os.path.basename(os.path.dirname(path))
    target = target_folder+folder_name
    if target[-1] is not "/":
        target += "/"

    print("Sampling to:",target)
    if not os.path.exists(target):
        os.makedirs(target)

    files = sorted(os.listdir(path))
    files = fnmatch.filter(files, '*.jpg')

    L = len(files)
    T = float(L) / float(N)

    sampled_files = []
    for i in range(0,N):
        v = int( i*T + T/2.0 )
        file = files[v]

        src = path + file
        dst = target + file

        copyfile(src, dst)

        sampled_files.append(dst)
    return sampled_files

def sample_N_from_folderS(path_base, folders, N, target_folder):
    for f in folders:
        p = path_base + f
        sample_N_from_folder(p, N, target_folder)

path_base = "/home/ekmek/intership_project/video_parser/_videos_to_test/RuzickaDataset/input/"
folders = ["S1000025_5fps","S1000039_5fps","S1000051_5fps","S1000010_5fps","S1000028_5fps","S1000040_5fps","S1000044_5fps","S1000021_5fps","S1000038_5fps","S1000041_5fps","S1000046_5fps"]
samples_folder = "/home/ekmek/intership_project/video_parser/_videos_to_test/RuzickaDataset/samples/"

sample_N_from_folderS(path_base, folders, 10, samples_folder)