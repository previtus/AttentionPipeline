# Used to process data saved from the individual runs as .csv outputs with numbers of crops in both attention and evaluation
# the format of these files is:
"""

S1000010_5fps_Splits1to2_attention;2;2;2;2 ... (will be always the same actually)
S1000010_5fps_Splits1to2_evaluation;8;7;6; ... (varies over time)
S1000010_5fps_Splits1to2_max_evaluation;8

We want to produce a histogram showing the distribution of how many crops it took to evaluate.
Evaluation only and evaluation with attention (the true value) // as well as comparison with how much it could have taken.

Graph 1: histogram of attention + evaluation
Graph 2: plot of attention + evaluation VS max evaluation

Multiple experiments in one?
Multigraph 1:
    I would like to compare various settings in the same video file
    (lets say "S1000010_5fps" with "1to2", "2to4" and "2to6")

Multigraph 2:
    I would like to compare different videos with the same settings
    (lets say "2to4" but over all the video files)

"""

##################### LOADING METHODS

import os, fnmatch

global crop_numbers_object

def load_from_csv(folder_path):
    files = sorted(os.listdir(folder_path))
    files = fnmatch.filter(files, '*.csv')

    crop_numbers_object = {}
    for filename in files:
        with open(folder_path+filename, 'r') as fp:
            read_lines = fp.readlines()
            read_lines = [line.rstrip('\n') for line in read_lines] # cut lines
            read_lines = [line.split(";") for line in read_lines] # cut ; items

            string_id = read_lines[0][0][0:-len("_attention")]
            # we saved: "S1000010_5fps_Splits1to2" -> which we can use to get name and setting choice
            name, _, setting = string_id.split("_")
            setting = setting[0+len("Splits"):]

            #print(name, setting)

            if setting not in crop_numbers_object:
                crop_numbers_object[setting] = {}
                crop_numbers_object[setting]["datasets"] = []
                crop_numbers_object[setting]["arrays"] = {}

            crop_numbers_object[setting]["datasets"].append(name)

            att_crops = read_lines[0][1] #redundantly saved
            max_crops = read_lines[2][1]
            crop_numbers_object[setting]["att_crops"] = int(att_crops)
            crop_numbers_object[setting]["max_crops"] = int(max_crops)

            array_with_evaluations = read_lines[1][1:]
            array_with_evaluations = [int(i) for i in array_with_evaluations]

            #crop_numbers_object[setting]["arrays"].append(array_with_evaluations)
            crop_numbers_object[setting]["arrays"][name] = array_with_evaluations


    for setting in crop_numbers_object.keys():
        print("Setting",setting, ":", list(crop_numbers_object[setting].keys()), "with", len(crop_numbers_object[setting]["datasets"]), len(crop_numbers_object[setting]["arrays"]))

    return crop_numbers_object

def get_vector_of_crop_numbers(include_evaluations, include_attentions, datasets, setting, crops_times_const=1.0, flip_to_wild_estimate_of_fps=False):
    """
    :param include_evaluations: True / False
    :param include_attentions: True / False
    :param datasets: array of datasets, subset from ['S1000010', 'S1000021', ..., 'S1000051']
    :param setting: for example "1to2" etc
    :return: single vector of all of these combined
    """
    attention_add = 0
    if include_attentions:
        attention_add = crop_numbers_object[setting]["att_crops"]
    evaluation_multi = 1
    if not include_evaluations:
        evaluation_multi = 0

    max_possible = crop_numbers_object[setting]["max_crops"]
    accumulated_vector = []
    for d in datasets:
        arr = crop_numbers_object[setting]["arrays"][d]

        if not flip_to_wild_estimate_of_fps:
            arr = [crops_times_const*(evaluation_multi*i + attention_add) for i in arr]
        else:
            arr = [1.0 / (crops_times_const * (evaluation_multi * i + attention_add)) for i in arr]
        #print(d, arr)

        accumulated_vector += arr

    return accumulated_vector, max_possible

path = "/home/ekmek/intership_project/video_parser_v1/_videos_to_test/RuzickaDataset/output/Histograms/"
crop_numbers_object = load_from_csv(path)

##################### VISUALIZATION METHODS
import numpy as np
import matplotlib, os
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.backends.backend_pdf
from matplotlib.ticker import MaxNLocator


def boxplot(data, title='', y_min=0.0, y_max=1.0, legend_on=True, notch=True):
    ''' Plot box plot / whisker graph from data.'''
    plt.figure(figsize=(5, 8))
    axes = plt.axes()
    axes.yaxis.set_major_locator(ticker.MultipleLocator(np.abs(y_max-y_min)/10.0))
    axes.yaxis.set_minor_locator(ticker.MultipleLocator(np.abs(y_max-y_min)/100.0))

    meanpointprops = dict(linewidth=1.0)
    boxplot = plt.boxplot(data, notch=notch, showmeans=True, meanprops=meanpointprops)

    plt.xticks([])

    if (legend_on):
        boxplot['medians'][0].set_label('median')
        boxplot['means'][0].set_label('mean')
        boxplot['fliers'][0].set_label('outlayers')
        # boxplot['boxes'][0].set_label('boxes')
        # boxplot['whiskers'][0].set_label('whiskers')
        boxplot['caps'][0].set_label('caps')

        axes.set_xlim([0.7, 1.7])

        plt.legend(numpoints = 1)

    #zoomOutY(axes,factor=0.1)
    axes.set_title(title)
    plt.show()

def show_row_multiple_datasets(datasets, setting, sort_by_means=True, show=True, overwrite_order=None, fix_y=None, crops_times_const=1.0, viz_data_type=''):
    plt.rcParams["figure.figsize"] = [16, 9]

    f, axes = plt.subplots(1, len(datasets), sharey=True)
    if viz_data_type=='time':
        plt.suptitle('Frame evaluation estimate, with setting "'+str(setting)+'"')
    elif viz_data_type=='fps':
        plt.suptitle('FPS rough estimate, with setting "'+str(setting)+'"')
    else:
        plt.suptitle('Number of crops over datasets, with setting "'+str(setting)+'"')
    f.subplots_adjust(wspace=0)
    if fix_y is not None:
        plt.ylim(fix_y[0], fix_y[1])

    indices = range(0,len(datasets))

    if sort_by_means and overwrite_order is not None:
        indices = []
        means = []
        for i,d in enumerate(datasets):
            v, _ = get_vector_of_crop_numbers(True, True, [d], setting)
            means.append( np.mean(v) )
            indices.append( i )

        indices = [x for _, x in sorted(zip(means, indices))]

    if overwrite_order is not None:
        indices = overwrite_order

    last = None
    last_maxhandler = None
    l = 0
    for i in indices:
        d = datasets[i]
        if viz_data_type == 'fps':
            v, max_possible = get_vector_of_crop_numbers(True, True, [d], setting, crops_times_const, flip_to_wild_estimate_of_fps=True)
        else:
            v, max_possible = get_vector_of_crop_numbers(True, True, [d], setting, crops_times_const)

        ax = axes[l]
        last = ax.boxplot(v, showmeans=True)
        ax.set_title(d)

        if viz_data_type=='time':
            mean_time = np.mean(v) # average of (numbers of active crops in a frame * time to evaluate a crop)
            fps_wild_estimate = 1.0 / mean_time

            ax.set(xlabel=str(round(fps_wild_estimate, 2))+'fps')
        l+=1

        if viz_data_type == 'fps':
            last_maxhandler = ax.plot([1], [1.0/(crops_times_const*max_possible)], marker='v', markersize=10, color="red")
        else:
            last_maxhandler = ax.plot([1], [crops_times_const*max_possible], marker='v', markersize=10, color="red")


    if (True):

        ax.legend([last['medians'][0], last['means'][0], last['fliers'][0], last['caps'][0], last_maxhandler[0]],
                  ['median', 'mean', 'outlayers', 'caps', 'all crops'], bbox_to_anchor=(1.04,1), loc="upper left")

    name = "plot_"
    if viz_data_type=='time':
        axes[0].set(ylabel='seconds per frame')
        name += "time"
    elif viz_data_type == 'fps':
            axes[0].set(ylabel='estimated fps')
            name += "fps"
    else:
        axes[0].set(ylabel='number of active crops')
        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        name += "crops"

    #frame1 = plt.gca()
    #frame1.axes.xaxis.set_ticklabels([])
    #axes[1].set(xlabel='')

    plt.savefig(name+setting+'.png', bbox_inches='tight')
    if show:
        plt.show()


all_datasets = crop_numbers_object["1to2"]["datasets"]
selected_datasets = ["S1000038", "S1000046"]
setting = "1to2"


v, _ = get_vector_of_crop_numbers(True, True, all_datasets, "2to4")
#print("min-avg-max:", np.min(v), np.mean(v), np.max(v))
max_y = 40
max_y = np.max(v) * 1.1
show_row_multiple_datasets(all_datasets, "1to2", False, False, None, [0, max_y])
show_row_multiple_datasets(all_datasets, "1to3", False, False, None, [0, max_y])
show_row_multiple_datasets(all_datasets, "2to3", False, False, None, [0, max_y])
show_row_multiple_datasets(all_datasets, "2to4", False, False, None, [0, max_y])


crops_times_const = 0.0401344 ### depends on Bridges load?
#crops_times_const = 0.0345


v, _ = get_vector_of_crop_numbers(True, True, all_datasets, "2to4", crops_times_const)
max_y = np.max(v) * 1.1

show_row_multiple_datasets(all_datasets, "1to2", False, False, None, [0, max_y], crops_times_const=crops_times_const, viz_data_type='time')
show_row_multiple_datasets(all_datasets, "1to3", False, False, None, [0, max_y], crops_times_const=crops_times_const, viz_data_type='time')
show_row_multiple_datasets(all_datasets, "2to3", False, False, None, [0, max_y], crops_times_const=crops_times_const, viz_data_type='time')
show_row_multiple_datasets(all_datasets, "2to4", False, False, None, [0, max_y], crops_times_const=crops_times_const, viz_data_type='time')


v, _ = get_vector_of_crop_numbers(True, True, all_datasets, "1to2", crops_times_const, flip_to_wild_estimate_of_fps=True)
#print("min-avg-max:", np.min(v), np.mean(v), np.max(v))
max_y = np.max(v) * 1.1

show_row_multiple_datasets(all_datasets, "1to2", False, False, None, [0, max_y], crops_times_const=crops_times_const, viz_data_type='fps')
show_row_multiple_datasets(all_datasets, "1to3", False, False, None, [0, max_y], crops_times_const=crops_times_const, viz_data_type='fps')
show_row_multiple_datasets(all_datasets, "2to3", False, False, None, [0, max_y], crops_times_const=crops_times_const, viz_data_type='fps')
show_row_multiple_datasets(all_datasets, "2to4", False, False, None, [0, max_y], crops_times_const=crops_times_const, viz_data_type='fps')
