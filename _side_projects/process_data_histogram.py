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

def get_vector_of_crop_numbers(include_evaluations, include_attentions, datasets, setting):
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

    accumulated_vector = []
    for d in datasets:
        arr = crop_numbers_object[setting]["arrays"][d]

        arr = [evaluation_multi*i + attention_add for i in arr]
        #print(d, arr)

        accumulated_vector += arr

    return accumulated_vector

path = "/home/ekmek/intership_project/video_parser/_videos_to_test/RuzickaDataset/output/Histograms/"
crop_numbers_object = load_from_csv(path)


all_datasets = crop_numbers_object["1to2"]["datasets"]
selected_datasets = ["S1000038", "S1000046"]
setting = "1to2"

v = get_vector_of_crop_numbers(True, True, selected_datasets, setting)

print(len(v), v[0:10], "...")
