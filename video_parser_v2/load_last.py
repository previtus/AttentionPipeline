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

    #print("Loading the last experiment")
    history = load_whole_history_and_settings(load_folder)
    history.settings.render_folder_name = save_folder

    return history


def plot_history(history, just_show):
    history.plot_and_save(show_instead_of_saving=just_show)
    print("Done---")


    #folder = "/home/ekmek/intership_project/video_parser_v2/__Renders/____OnSERVERrunsClient/S1000040_5fps_1to2/"
#load_last(folder,folder)