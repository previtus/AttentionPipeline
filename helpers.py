import os

def make_dir_if_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)