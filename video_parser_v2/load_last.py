import History
import pickle

def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
        return obj

def load_whole_history_and_settings(folder=''):
    settings = load_object(folder + "settings_last.pkl")
    history = load_object(folder + "history_last.pkl")
    history.settings = settings

    return history

def load_last(load_folder, save_folder):

    print("Loading the last experiment")
    history = load_whole_history_and_settings(load_folder)
    history.settings.render_folder_name = save_folder

    history.plot_and_save()
    print("Done---")

folder = "/home/ekmek/intership_project/video_parser_v2/__Renders/March28_saving/"
load_last(folder,folder)