from results_plotting.multiple_graph_plots import *
from load_last import load_last, plot_history
import re, os
def load_histories(folders):
    histories = []
    for folder in folders:
        history = load_last(folder, folder)
        # ps careful about the history.settings.render_folder_name = save_folder
        histories.append(history)

    return histories

def select_all_subdirectories(root_folder):
    folders = [x[0] for x in os.walk(root_folder)]
    folders = sorted(folders)
    return folders[1:] # first is self

def select_subdirectories(root_folder, pattern):
    folders = select_all_subdirectories(root_folder)
    R = re.compile(pattern)
    filtered = [folder+"/" for folder in folders if R.match(folder)]
    names = [folder.split("/")[-2] for folder in filtered]
    return filtered, names
