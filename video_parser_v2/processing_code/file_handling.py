import os


def is_non_zero_file(fpath):
    # thanks https://stackoverflow.com/a/15924160
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
