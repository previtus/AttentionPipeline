from voc_eval import voc_eval, parse_rec, voc_eval_twoArrs
from mark_frame_with_bbox import annotate_image_with_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np
import os

"/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/Custom 4k videos/"

# PEViD
#predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/PEViD_clean/"
predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/PEViD_full_take3/"
#predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/PEViD_clean-(wrong-run)/" # 0.912
# without cleanSet_Exchanging_bags_day_outdoor_5_1to3 ? == 0.94
globally_saved_annotations = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/PEViD_full_1_copy___PEViD_just_annotations/GroundTruthsForPEViD_voc_format/"

# Custom 4K
#predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/Custom_4k_videos/1to2_over20/"
#predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/Custom_4k_videos/1to3_over20/"
#predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/Custom_4k_videos/2to4_over20/"
#predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/Custom_4k_videos/2to6_over20/"
predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/Custom_4k_videos/1to1_onlyNMS/"
predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/PEViD_full_alt_sizes/allcrops_1to3_justNMS/"
#predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/PEViD_full_alt_sizes/1to1_postproc/"
#globally_saved_annotations = "/media/ekmek/VitekDrive_I/2017-18_CMU internship, part 1, Fall 2017/4K_DATASET_REC/annotations/samples/"

overlap_threshold = 0.25 # AP 0.5, AP 0.75, AP 0.25 check

assert predictions_folder[-1] == "/"
assert globally_saved_annotations[-1] == "/"

# Accuracy
# predictions_folder contains folders with predictions (annotbboxes.txt and annotnames.txt)
#  and also a link to the ground truth folder name in settings.txt
# globally_saved_annotations contains all name specific folders with many .xml files for each frame

folders = [name for name in os.listdir(predictions_folder) if os.path.isdir(os.path.join(predictions_folder, name))]

x_aps = []
x_labels = []

for folder in folders:
    #print("FOLDER", folder)
    setting_file = predictions_folder+folder+"/settings.txt"
    source_folder_name = ""

    with open(setting_file) as file:
        for line in file:
            if "INPUT_FRAMES" in line:
                #print(line) # ... PEViD_UHD_annot/Stealing_day_indoor_3/",
                s = line.split("/")
                source_folder_name = s[-2] #Stealing_day_indoor_3

                break
    # now we have predictions in folder and the gt annotations in folder by name source_folder_name
    gt_annotation_folder = globally_saved_annotations + source_folder_name+"/"


    imagesetfile = predictions_folder+folder+"/annotnames.txt"
    predictions_file  = predictions_folder+folder+"/annotbboxes.txt"


    #print("Predictions from", predictions_file)
    #print("GT from", gt_annotation_folder)
    rec, prec, ap = voc_eval(predictions_file,gt_annotation_folder,imagesetfile,'person', overlap_threshold)

    print(folder," =ap=> ", ap)

    x_aps.append(ap)
    x_labels.append(folder)


# optional sort by mAP:

sorted_indices = np.argsort(x_aps)
#x_aps = [ x_aps[id] for id in sorted_indices ]
#x_labels = [ x_labels[id] for id in sorted_indices ]

fig = plt.figure()
plt.title("AP over folder, AP"+str(overlap_threshold)+": "+str(np.mean(x_aps)))
plt.plot(x_aps, linestyle='-', marker='o')
plt.xticks(np.arange(len(x_aps)), x_labels, rotation='vertical')
plt.ylim(0.0, 1.0)

fig.subplots_adjust(bottom=0.35)
plt.show()
