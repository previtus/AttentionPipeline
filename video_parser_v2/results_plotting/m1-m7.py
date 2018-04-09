from results_plotting.multiple_graph_plots import *
from results_plotting.multiple_graph_loading import *



# THIS IS DIRECTORY SPECIFIC
# IN OUR CASE THESE WORK: tuples of (root_folder, pattern, names function(applies for all above it) )
root_folder = "/home/ekmek/__RESULTS_and_MEASUREMENTS/M_experiment/_m1-m3_from local/"
pattern = ".*/m1.*"
pattern = ".*/m2.*"
pattern = ".*/m3.*"
# names = [name[13:21] for name in names]
# pattern = ".*" # all
root_folder = "/home/ekmek/__RESULTS_and_MEASUREMENTS/M_experiment/_m4-m7_fully  serverside/"
pattern = ".*/m4.*"
pattern = ".*/m5.*"
# pattern = ".*/m6.*" # nvm
# pattern = ".*/m7.*" # names = [name[3:11] for name in names]

# pattern = ".*" # all
# names = [name[23:32] for name in names]


root_folder = "/home/ekmek/__RESULTS_and_MEASUREMENTS/M_experiment/_m1-m3_from local/"
pattern = ".*/m3.*"

folders, names = select_subdirectories(root_folder, pattern)

names = [name[13:21] for name in names]

histories = load_histories(folders)

for i, h in enumerate(histories):
    print("RUN:", h.settings.RUN_NAME, "    name:", names[i],
          "  ServersLimit:", h.settings.final_evaluation_limit_servers, "+1 att",
          "  Splits", str(h.settings.attention_horizontal_splits) + "to" + str(h.settings.horizontal_splits))

    plt = plt

    one_stackedbar(h, plt, show_instead_of_saving=True, column=i + 1)

if True: # alter this as needed
    plt.ylim(0.0, 1.0)

#plt.title("Experiment m1, [040_5fps_1to2], on servers = 2-9")
#plt.title("Experiment m2, [040_5fps_2to4], on servers = 2-16")
plt.title("Experiment m3, [8k video_2to4], on servers = 2-16")
#plt.title("Experiment m4, serverside [040_5fps_2to4], on servers = 2-16")
#plt.title("Experiment m5, serverside [8k video_2to6], on servers = 2-16")
#plt.title("Experiment m7, FullRuns, on all 16 serves")
plt.ylabel("Time (s)")
plt.xlabel("run with different setting")
plt.xticks(range(1, len(histories) + 1), names)

plt.show()

# load multiple

# put multiple as the stackedbars with errors

# put one(few) item(s) from mutiple as plotboxes