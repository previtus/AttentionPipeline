from results_plotting.multiple_graph_plots import *
from results_plotting.multiple_graph_loading import *

# custom selection of folders:
"""
- all done on 16 workers (see what it can maximally do with different datasets + settings)
    - 8k video, 2to4 local
    - 8k video, 2to6 serverside
    - dense (almost always active) 2to4 video
    - dense 1to2 video
    - (mid / low density)
    - serverside / local side

/home/ekmek/__RESULTS_and_MEASUREMENTS/_m1-m3_from local/m2_MASS_1p15_all16servers_040_5fps_2to4
/home/ekmek/__RESULTS_and_MEASUREMENTS/_m4-m7_fully  serverside/m4_040serversideclient_16servers
/home/ekmek/__RESULTS_and_MEASUREMENTS/_m4-m7_fully  serverside/m5_8k_serversideclient_16servers

"""


specific_folders = [
    "/home/ekmek/__RESULTS_and_MEASUREMENTS/_m1-m3_from local/m2_MASS_1p15_all16servers_040_5fps_2to4/",
    "/home/ekmek/__RESULTS_and_MEASUREMENTS/_m4-m7_fully  serverside/m4_040serversideclient_16servers/",
    "/home/ekmek/__RESULTS_and_MEASUREMENTS/_m4-m7_fully  serverside/m5_8k_serversideclient_16servers/",
    "/home/ekmek/__RESULTS_and_MEASUREMENTS/_m4-m7_fully  serverside/m7_021_full_frames1min_serversideclient_all/",
    "/home/ekmek/__RESULTS_and_MEASUREMENTS/_m4-m7_fully  serverside/m7_038_full_frames1min_verb1_serversideclient_all/",
    "/home/ekmek/__RESULTS_and_MEASUREMENTS/_m4-m7_fully  serverside/m7_041_full_frames1min_verb1_serversideclient_all/",
    "/home/ekmek/__RESULTS_and_MEASUREMENTS/_m4-m7_fully  serverside/m7_051_full_frames1min_verb1_serversideclient_all/"
]
names = ["040 local 2to4", "040 serverside 2to4", "8k serverside 2to6",
         "021 serverside 2to4", "038 serverside 2to4", "041 serverside 2to4", "051 serverside 2to4"]


histories = load_histories(specific_folders)

for i,h in enumerate(histories):
    print("RUN:", h.settings.RUN_NAME,"    name:", names[i],
          "  ServersLimit:", h.settings.final_evaluation_limit_servers,"+1 att",
          "  Splits", str(h.settings.attention_horizontal_splits)+"to"+str(h.settings.horizontal_splits))

    plt = plt

    one_stackedbar(h,plt,show_instead_of_saving=True, column=i+1)

if False: # alter this as needed
    plt.ylim(0.0, 0.50)

plt.title("Comparison of all 16 server runs")
plt.ylabel("Average time over whole run (s)")
plt.xlabel("run with different setting")
plt.xticks(range(1,len(histories)+1), names)

plt.show()



