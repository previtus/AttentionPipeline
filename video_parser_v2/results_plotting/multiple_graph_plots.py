from load_last import load_last, plot_history
import re, os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from cycler import cycler



def timing_per_frame_plot_stackedbar(self,show_instead_of_saving):
    y_limit = True # limit the plot to 0 - 1sec ?

    # prepare lists:
    IO_loads = list(self.IO_loads.values())
    IO_saves = list(self.IO_saves.values())
    if self.settings.precompute_attention_evaluation:
        AttWait = list(self.times_attention_evaluation_waiting.values())
        attention_name = "AttWait"
    else:
        AttWait = list(self.times_attention_evaluation_processing_full_frame.values())
        attention_name = "AttEval"

    ##FinalCutEval = list(self.times_final_evaluation_processing_full_frame.values())
    FinalCut = list(self.IO_EVAL_cut_crops_final.values())


    # these are ~ [[0...N], [0...N]] N values across N servers
    FinalEncode = list(self.times_final_evaluation_processing_per_worker_encode.values())
    FinalDecode = list(self.times_final_evaluation_processing_per_worker_decode.values())
    FinalTransfer = list(self.times_final_evaluation_processing_per_worker_transfers.values())
    FinalEval = list(self.times_final_evaluation_processing_per_worker_eval.values())

    # flatten lists
    FinalTransfer = [np.mean(item) for item in FinalTransfer]
    FinalEval = [np.mean(item) for item in FinalEval]
    FinalEncode = [np.mean(item) for item in FinalEncode]
    FinalDecode = [np.mean(item) for item in FinalDecode]

    ###FinalTransfer = list(self.times_final_full_frame_avg_transfer.values())
    ###FinalEval = np.array(FinalCutEval) - np.array(FinalCut) - np.array(FinalTransfer)

    postprocess = list(self.postprocess.values())

    IO_loads = np.array(IO_loads)
    IO_saves = np.array(IO_saves)
    if len(IO_loads) > len(IO_saves):
        IO_loads = IO_loads[0:len(IO_saves)]

    AttWait = np.array(AttWait)
    FinalCut = np.array(FinalCut)
    postprocess = np.array(postprocess)

    N = len(IO_loads)
    ind = np.arange(N)
    width = 0.35

    plt.title("Per frame analysis - stacked bar")
    plt.ylabel("Time (s)")
    plt.xlabel("Frame #num")

    p1 = plt.bar(ind, IO_loads+IO_saves, width, color='yellow') # yerr=stand deviation
    bottom = IO_saves + IO_loads
    p2 = plt.bar(ind, AttWait, width, bottom=bottom, color='blue')
    bottom += AttWait
    p3a = plt.bar(ind, FinalCut, width, bottom=bottom, color='lightcoral')
    bottom += FinalCut
    p3b = plt.bar(ind, FinalEncode, width, bottom=bottom, color='navajowhite')
    bottom += FinalEncode
    p4a = plt.bar(ind, FinalDecode, width, bottom=bottom, color='burlywood')
    bottom += FinalDecode
    p4b = plt.bar(ind, FinalTransfer, width, bottom=bottom, color='magenta')
    bottom += FinalTransfer
    p4c = plt.bar(ind, FinalEval, width, bottom=bottom, color='red')
    bottom += FinalEval
    p5 = plt.bar(ind, postprocess, width, bottom=bottom, color='green')
    bottom += postprocess

    if y_limit:
        ymin, ymax = plt.ylim()
        if ymax < 1.0:
            plt.ylim(0.0, 1.0)

    plt.legend((p1[0], p2[0], p3a[0], p3b[0], p4a[0], p4b[0], p4c[0], p5[0]),
           ('IO load&save', attention_name, 'Crop', 'Enc', 'Dec', 'Transfer', 'Eval', 'postprocess'))

    if show_instead_of_saving:
        plt.show()
    else:
        save_path = self.settings.render_folder_name + "stacked.png"
        plt.savefig(save_path, dpi=120)

    plt.clf()
    return 0

def one_stackedbar(self,plt,show_instead_of_saving, column):
    y_limit = True # limit the plot to 0 - 1sec ?

    # prepare lists:
    IO_loads = list(self.IO_loads.values())
    IO_saves = list(self.IO_saves.values())
    if self.settings.precompute_attention_evaluation:
        AttWait = list(self.times_attention_evaluation_waiting.values())
        attention_name = "AttWait"
    else:
        AttWait = list(self.times_attention_evaluation_processing_full_frame.values())
        attention_name = "AttEval"

    ##FinalCutEval = list(self.times_final_evaluation_processing_full_frame.values())
    FinalCut = list(self.IO_EVAL_cut_crops_final.values())


    # these are ~ [[0...N], [0...N]] N values across N servers
    FinalEncode = list(self.times_final_evaluation_processing_per_worker_encode.values())
    FinalDecode = list(self.times_final_evaluation_processing_per_worker_decode.values())
    FinalTransfer = list(self.times_final_evaluation_processing_per_worker_transfers.values())
    FinalEval = list(self.times_final_evaluation_processing_per_worker_eval.values())

    # flatten lists
    # up to 16 workers - we should ideally look at the most delaying one
    FinalTransfer = np.array(FinalTransfer)
    FinalEval = np.array(FinalEval)
    FinalEncode = np.array(FinalEncode)
    FinalDecode = np.array(FinalDecode)

    # this is the fix:
    cummulative = [[]]*len(FinalTransfer)
    for i in range(0,len(FinalTransfer)):
        #print(len(FinalTransfer[i]),len(FinalEval[i]),len(FinalEncode[i]),len(FinalDecode[i]))
        cummulative[i] = [[]]*len(FinalTransfer[i])
        for j in range(0,len(FinalTransfer[i])):
            cummulative[i][j] = FinalTransfer[i][j] + FinalEval[i][j] + FinalEncode[i][j] + FinalDecode[i][j]

    indices = [np.argmax(item) for item in cummulative]

    FinalTransferT = [[]] * len(FinalTransfer)
    FinalEvalT = [[]] * len(FinalTransfer)
    FinalEncodeT = [[]] * len(FinalTransfer)
    FinalDecodeT = [[]] * len(FinalTransfer)

    for i in range(0, len(FinalTransfer)):
        ind = indices[i]

        FinalTransferT[i] = FinalTransfer[i][ind]
        FinalEvalT[i] = FinalEval[i][ind]
        FinalEncodeT[i] = FinalEncode[i][ind]
        FinalDecodeT[i] = FinalDecode[i][ind]

    FinalTransfer = FinalTransferT
    FinalEval = FinalEvalT
    FinalEncode = FinalEncodeT
    FinalDecode = FinalDecodeT

    ###FinalTransfer = list(self.times_final_full_frame_avg_transfer.values())
    ###FinalEval = np.array(FinalCutEval) - np.array(FinalCut) - np.array(FinalTransfer)

    postprocess = list(self.postprocess.values())

    #print("shape of postprocess is:")
    #print(len(postprocess))
    #print("shape of FinalTransfer is:")
    #print(len(FinalTransfer))


    IO_loads = np.array(IO_loads)
    IO_saves = np.array(IO_saves)
    if len(IO_loads) > len(IO_saves):
        IO_loads = IO_loads[0:len(IO_saves)]

    AttWait = np.array(AttWait)
    FinalCut = np.array(FinalCut)
    postprocess = np.array(postprocess)

    width = 0.35
    ind = (column)

    p0 = plt.bar(ind, np.mean(IO_loads), width,color='orange')  # yerr=stand deviation
    bottom = np.mean(IO_loads)
    p1 = plt.bar(ind, np.mean(IO_saves), width, bottom=bottom, color='yellow')  # yerr=stand deviation
    bottom += np.mean(IO_saves)
    p2 = plt.bar(ind, np.mean(AttWait), width, bottom=bottom, color='blue')
    bottom += np.mean(AttWait)
    p3a = plt.bar(ind, np.mean(FinalCut), width, bottom=bottom, color='lightcoral')
    bottom += np.mean(FinalCut)
    p3b = plt.bar(ind, np.mean(FinalEncode), width, bottom=bottom, color='navajowhite')
    bottom += np.mean(FinalEncode)
    p4a = plt.bar(ind, np.mean(FinalDecode), width, bottom=bottom, color='burlywood')
    bottom += np.mean(FinalDecode)
    p4b = plt.bar(ind, np.mean(FinalTransfer), width, bottom=bottom, color='magenta')
    bottom += np.mean(FinalTransfer)
    p4c = plt.bar(ind, np.mean(FinalEval), width, bottom=bottom, color='red')
    bottom += np.mean(FinalEval)
    p5 = plt.bar(ind, np.mean(postprocess), width, bottom=bottom, color='green')
    bottom += np.mean(postprocess)

    """
    p1 = plt.bar(ind, np.mean(IO_loads+IO_saves), width, yerr=np.std(IO_loads+IO_saves), color='yellow') # yerr=stand deviation
    bottom = np.mean(IO_saves + IO_loads)
    p2 = plt.bar(ind, np.mean(AttWait), width, yerr=np.std(AttWait), bottom=bottom, color='blue')
    bottom += np.mean(AttWait)
    p3a = plt.bar(ind, np.mean(FinalCut), width, yerr=np.std(FinalCut), bottom=bottom, color='lightcoral')
    bottom += np.mean(FinalCut)
    p3b = plt.bar(ind, np.mean(FinalEncode), width, yerr=np.std(FinalEncode), bottom=bottom, color='navajowhite')
    bottom += np.mean(FinalEncode)
    p4a = plt.bar(ind, np.mean(FinalDecode), width, yerr=np.std(FinalDecode), bottom=bottom, color='burlywood')
    bottom += np.mean(FinalDecode)
    p4b = plt.bar(ind, np.mean(FinalTransfer), width, yerr=np.std(FinalTransfer), bottom=bottom, color='magenta')
    bottom += np.mean(FinalTransfer)
    p4c = plt.bar(ind, np.mean(FinalEval), width, yerr=np.std(FinalEval), bottom=bottom, color='red')
    bottom += np.mean(FinalEval)
    p5 = plt.bar(ind, np.mean(postprocess), width, yerr=np.std(postprocess), bottom=bottom, color='green')
    bottom += np.mean(postprocess)

    """

    if column == 1:

        # Describe only once
        plt.legend((p0[0], p1[0], p2[0], p3a[0], p3b[0], p4a[0], p4b[0], p4c[0], p5[0]),
           ('IO load', 'IO save', attention_name, 'Crop', 'Enc', 'Dec', 'Transfer', 'Eval', 'postprocess'))

        if y_limit:
            ymin, ymax = plt.ylim()
            if ymax < 1.0:
                plt.ylim(0.0, 1.0)


    return plt
