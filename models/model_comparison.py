from __future__ import print_function
# histories:
from helpers import *

crops_history_path = "/home/ekmek/saliency_tools/models/crops_history_3k.npy"
#crops_history_path = "/home/ekmek/saliency_tools/models/crops_history_6k.npy"
resized_history_path = "/home/ekmek/saliency_tools/models/resized_history.npy"
fullsize_history_path = "/home/ekmek/saliency_tools/models/fullsize_history.npy"

#fullsize_history_path = "/home/ekmek/saliency_tools/models/fullsize_history_B.npy"
#fullsize_history_path = "/home/ekmek/saliency_tools/models/fullsize_history_100ep.npy"


paths = [crops_history_path, resized_history_path, fullsize_history_path]
names = ["crops", "resized", "fullsize"]

histories_obj = [load_history(path) for path in paths]

histories = [histories_obj[0][0], histories_obj[1][0], histories_obj[2][0]]
parameters = ['loss', 'loss', 'loss']
parameters_val = ['clustered_mse', 'loss', 'loss']


visualize_histories(histories, names, parameters, parameters_val, custom_title="Model comparison")