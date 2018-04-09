from results_plotting.multiple_graph_plots import *
from results_plotting.multiple_graph_loading import *



# THIS IS DIRECTORY SPECIFIC
# IN OUR CASE THESE WORK: tuples of (root_folder, pattern, names function(applies for all above it) )
root_folder = "/home/ekmek/__RESULTS_and_MEASUREMENTS/N_experiment/"

#pattern = ".*/n2.*"
pattern = ".*/n3.*"

#names = [name[12:20] for name in names]

pattern = ".*/n1.*"
#root_folder = "/home/ekmek/__RESULTS_and_MEASUREMENTS/N_experiment/n1-pylon2_4k/"
pattern = ".*/n4.*"
# names = [name[13:21] for name in names]

pattern = ".*/n5local.*"
#names = [name[12:20]+'|'+name[31:] for name in names]


folders, names = select_subdirectories(root_folder, pattern)

folders = folders[1:]
names = names[1:]
#print(names)
#print(folders)

names = [name[12:20]+'|'+name[31:] for name in names]

#print(names)

histories = load_histories(folders)

for i, h in enumerate(histories):
    print("RUN:", h.settings.RUN_NAME, "    name:", names[i],
          "  ServersLimit:", h.settings.final_evaluation_limit_servers, "+1 att",
          "  Splits", str(h.settings.attention_horizontal_splits) + "to" + str(h.settings.horizontal_splits))

    plt = plt

    one_stackedbar(h, plt, show_instead_of_saving=True, column=i + 1)

if False: # alter this as needed
    plt.ylim(0.0, 1.0)

#plt.title("Experiment n1, [040 2to4] PYLON 2, on servers = 2-14, att 1-4")
#plt.title("Experiment n2, [8k video_2to6] PYLON 2, on servers = 2-14, att 1-4")
#plt.title("Experiment n3, [8k video_2to6] SCRATCH, on servers = 2-14, att 1 or 2")
#plt.title("Experiment n4, [4k 010 2to4] SCRATCH, on servers = 2-12, att 1 or 2")
plt.title("Experiment n5local, [4k 010 2to4] from local PC, on servers = 2-12, att 1 or 2")

plt.ylabel("Time (s)")
plt.xlabel("run with different setting")
plt.xticks(range(1, len(histories) + 1), names)
plt.tight_layout()

plt.show()