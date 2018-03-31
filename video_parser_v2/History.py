from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from cycler import cycler

class History(object):
    """
    Keeps interesting history such as which/how many crops were active, evaluation times, etc.
    Can also contain functions for plotting these into graphs.

    These objects have access to History:
        VideoCapture - can enter the full time it took for a frame (including every IO and postprocessing)
                     - also IO load
        Evaluation - can enter the evaluation time for attention + final evaluation stage;
                     this should also include how many crops were evaluated
                     (we are interested in performance per frame and also per crop)
                   - can separate FinalEval time with per server (worker) information
                     and AttentionEval and AttentionWait
                        AttentionEval = how much time it took to evaluate attention
                        AttentionWait = how long did we have to wait for it in the main loop (for precomputation)
                   - also IO cut crops

        AttentionModel - can enter the number of active crops in each frame

        Renderer - can enter number of bounding boxes in final image
                 - alse IO save (or IO render)

    """

    def __init__(self, settings):
        self.settings = settings
        self.frame_ticker = self.settings.render_history_every_k_frames

        # AttentionModel feed the number of active crops / all crops
        self.active_crops_per_frames = {}
        self.total_crops_per_frames = {}

        self.total_crops_during_attention_evaluation = {}

        # Evaluation and VideoCapture feed time for
        self.times_attention_evaluation_processing_full_frame = {}
        self.times_final_evaluation_processing_full_frame = {}

        # per server/worker information
        self.times_final_evaluation_processing_per_worker = {} # should have an array or times for each frame
        self.times_attention_evaluation_processing_per_worker = {}

        self.times_attention_evaluation_waiting = {}

        self.times_evaluation_each_loop = {}

        # Renderer
        self.number_of_detected_objects = {}

        # IO related
        self.IO_loads = {} # in VideoCapture
        self.IO_EVAL_cut_crops_attention = {} # in attention Evaluation
        self.IO_EVAL_cut_crops_final = {} # in final Evaluation
        self.IO_saves = {} # in Render

        # Postprocessing
        self.postprocess = {}  # in Render

        # is it worth it to measure each crops evaluation time? maybe not really

        self.frame_number = -1

        self.loop_timer = None

    def tick_loop(self, frame_number, force=False):
        # measures every loop
        if self.loop_timer is None:
            # start the measurements!

            self.loop_timer = timer()
            self.frame_number = 0
        else:
            last = self.loop_timer
            self.loop_timer = timer()
            t = self.loop_timer - last

            if self.settings.verbosity >= 2:
                print("Timer, loop timer t=", t)

            self.times_evaluation_each_loop[frame_number]=t

            self.end_of_frame(force)
            # we want to disregard graph plotting time:
            self.loop_timer = timer()

    def report_crops_in_attention_evaluation(self, number_of_attention_crops, frame_number):
        self.total_crops_during_attention_evaluation[frame_number] = number_of_attention_crops

    def report_attention(self, active, total, frame_number):
        self.active_crops_per_frames[frame_number] = active
        self.total_crops_per_frames[frame_number] = total

    def report_evaluation_whole_function(self, type, for_frame, frame_number):
        if type == 'attention':
            self.times_attention_evaluation_processing_full_frame[frame_number]=for_frame
        elif type == 'evaluation':
            self.times_final_evaluation_processing_full_frame[frame_number]=for_frame

    def report_evaluation_per_individual_worker(self, times, type, frame_number):
        if type == 'attention':
            # however for this one, we don't care for now
            # frame_number can be 'in future' when precomputing
            self.times_attention_evaluation_processing_per_worker[frame_number] = times
        elif type == 'evaluation':
            self.times_final_evaluation_processing_per_worker[frame_number] = times

    def report_evaluation_attention_waiting(self, time, frame_number):
        self.times_attention_evaluation_waiting[frame_number] = time

    def report_number_of_detected_objects(self, number_of_detected_objects, frame_number):
        self.number_of_detected_objects[frame_number]=number_of_detected_objects

    def report_skipped_final_evaluation(self, frame_number):
        # No active crop was detected - we can just skip the finner evaluation
        self.times_final_evaluation_processing_full_frame[frame_number]=0
        self.IO_EVAL_cut_crops_final[frame_number]=0
        self.postprocess[frame_number]=0
        self.number_of_detected_objects[frame_number]=0

    def report_IO_load(self, time, frame_number):
        self.IO_loads[frame_number] = time

    def report_IO_save(self, time, frame_number):
        self.IO_saves[frame_number] = time

    def report_postprocessing(self, time, frame_number):
        self.postprocess[frame_number] = time

    def report_IO_EVAL_cut_evaluation(self, type, time, frame_number):
        if type == 'attention':
            self.IO_EVAL_cut_crops_attention[frame_number]=time
        elif type == 'evaluation':
            self.IO_EVAL_cut_crops_final[frame_number]=time

    def end_of_frame(self, force=False):
        self.frame_ticker -= 1
        if self.frame_ticker <= 0 or force:

            print("History report!")
            self.frame_ticker = self.settings.render_history_every_k_frames

            self.plot_and_save()

    def plot_and_save(self):
        self.print_all_datalists()
        self.timing_per_frame_plot_stackedbar() ###
        self.timing_per_frame_plot_boxplot() ####

        for_attention_measure_waiting_instead_of_time = self.settings.precompute_attention_evaluation

        active_crops_per_frames = list(self.active_crops_per_frames.values())
        total_crops_per_frames = list(self.total_crops_per_frames.values())
        total_crops_during_attention_evaluation = list(self.total_crops_during_attention_evaluation.values())

        times_attention_evaluation_processing_full_frame = list(self.times_attention_evaluation_processing_full_frame.values())

        times_attention_evaluation_waiting = list(self.times_attention_evaluation_waiting.values())

        if for_attention_measure_waiting_instead_of_time:
            times_attention_evaluation_processing_full_frame = times_attention_evaluation_waiting
        times_final_evaluation_processing_full_frame = list(self.times_final_evaluation_processing_full_frame.values())

        times_evaluation_each_loop = list(self.times_evaluation_each_loop.values())

        # Renderer
        number_of_detected_objects = list(self.number_of_detected_objects.values())


        # plot all of the history graphs:

        style = "areas" # lines of areas
        color_change = True
        # red, yellow, blue, green, blue 2, orange, pink, grey, almost white
        color_list = ['#ec9980', '#f3f3bb', '#8fb6c9', 'accc7c', '#94ccc4', '#ecb475', '#f4d4e4', '#dcdcdc', '#fbfbfb']

        # speed performance:
        # - plot line for attention evaluation
        # - plot line for attention + final evaluations
        # - plot line for full frame

        plt.figure(1)
        plt.subplot(211)

        if color_change: plt.gca().set_prop_cycle(cycler('color', color_list))

        self.prepare_plot(plt,"Speed performance", "", "evaluation in sec") # frames

        attention_plus_evaluation = [sum(x) for x in zip(times_final_evaluation_processing_full_frame, times_attention_evaluation_processing_full_frame)]

        if style == "areas":
            x = range(len(times_attention_evaluation_processing_full_frame))
            plt.fill_between(x, 0, times_attention_evaluation_processing_full_frame, label="Attention")
            plt.fill_between(x, times_attention_evaluation_processing_full_frame, attention_plus_evaluation, label="Attention+Final")
            plt.fill_between(x, attention_plus_evaluation, times_evaluation_each_loop, label="Whole loop") # 34 and 35
        elif style == "lines":
            plt.plot(times_attention_evaluation_processing_full_frame, label="Attention")
            plt.plot(attention_plus_evaluation, label="Attention+Final")
            plt.plot(times_evaluation_each_loop, label="Whole loop")

        self.limit_y_against_outliers(plt, times_evaluation_each_loop)

        plt.legend(loc='upper left', shadow=True)

        # active crops
        # - plot line for active
        # - plot line for total

        plt.subplot(212)
        if color_change: plt.gca().set_prop_cycle(cycler('color', color_list))

        self.prepare_plot(plt, "", "frames", "number of crops") # Active crops (includes attention)

        # dict.items()

        # ! this is number in final evaluation
        active_plus_attention = [sum(x) for x in zip(active_crops_per_frames,
                                                         total_crops_during_attention_evaluation)]
        total_plus_attention = [sum(x) for x in zip(total_crops_per_frames,
                                                         total_crops_during_attention_evaluation)]

        if style == "areas":
            x = range(len(total_crops_per_frames))
            plt.fill_between(x, 0, active_plus_attention, label="Active")
            plt.fill_between(x, active_plus_attention, total_plus_attention, label="Total crops")
        elif style == "lines":
            plt.plot(active_plus_attention, label="Active")
            plt.plot(total_plus_attention, label="Total crops")

        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend(loc='upper left', shadow=True)


        #plt.show()
        save_path = self.settings.render_folder_name + "last_plot.png"
        plt.savefig(save_path, dpi=120)

        plt.clf()
        return 0

    def prepare_plot(self, plt, title, xlabel, ylabel):
        #plt.clf()

        # plt.gca().invert_yaxis()
        # plt.xticks(())
        # plt.yticks(())
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        #plt.legend(names, loc='best')

        return plt

    def limit_y_against_outliers(self, plt, data):

        max_value = np.max(data)
        most_below = np.percentile(data, 95)

        #print("max_value, most_below", max_value, most_below)
        #print("ratio", most_below / max_value)

        if (most_below / max_value) < 0.5: # usually close to 0.999
            # outliers might be significantly messing up the graph
            # max value is 2x what most of the data is...
            plt.ylim(ymax=most_below, ymin=0)

    def timing_per_frame_plot_idea(self):
        """
        Idea of this plotting is:

        (yellow+orange)   |  (light blue)  | (light red)                | (green)

        IO load, IO save  | Attention Wait | Final cut + eval           | Postprocessing
                                             [[cut] + per server] total
        """

        # IO load, IO save
        self.IO_loads
        self.IO_saves

        # Attention Wait
        self.times_attention_evaluation_waiting

        # Final cut + eval
        self.times_final_evaluation_processing_full_frame
        # alternatively: [[cut] + per server] total
        self.IO_EVAL_cut_crops_final
        self.times_final_evaluation_processing_per_worker

        # Postprocessing
        self.postprocess


        return 0

    def timing_per_frame_plot_boxplot(self):
        """
        (yellow+orange)   |  (light blue)  | (light red)                | (green)
        IO load, IO save  | Attention Wait | Final cut + eval           | Postprocessing
                                             [[cut] + per server] total
        """

        # 1.) into lists
        # IO load, IO save
        IO_loads = list(self.IO_loads.values())
        IO_saves = list(self.IO_saves.values())

        # Attention Wait
        if self.settings.precompute_attention_evaluation:
            AttWait = list(self.times_attention_evaluation_waiting.values())
            attention_name = "AttWait"
        else:
            AttWait = list(self.times_attention_evaluation_processing_full_frame.values())
            attention_name = "AttEval"

        # Final cut + eval
        FinalCutEval = list(self.times_final_evaluation_processing_full_frame.values())
        FinalCut = list(self.IO_EVAL_cut_crops_final.values())
        FinalEval = np.array(FinalCutEval) - np.array(FinalCut)

        # alternatively: [[cut] + per server] total
        #Cuts = list(self.IO_EVAL_cut_crops_final.values())
        #FinalPerWorker = list(self.times_final_evaluation_processing_per_worker.values())

        # Postprocessing
        postprocess = list(self.postprocess.values())

        #data = [IO_loads, IO_saves, AttWait, FinalCutEval, postprocess]
        data = [IO_loads, IO_saves, AttWait, FinalCut, FinalEval, postprocess]
        # multiple box plots on one figure

        plt.title("Per frame analysis - box plot")
        plt.ylabel("Time (s)")
        plt.xlabel("Stages")

        plt.boxplot(data)
        #plt.xticks(range(1,6), ['IO_loads', 'IO_saves', attention_name, 'FinalCutEval', 'postprocess'])
        plt.xticks(range(1,7), ['IO_loads', 'IO_saves', attention_name, 'FinalCut', 'FinalEval', 'postprocess'])
        # https://matplotlib.org/gallery/statistics/boxplot_demo.html

        save_path = self.settings.render_folder_name + "boxplot.png"
        plt.savefig(save_path, dpi=120)

        #plt.show()
        plt.clf()
        return 0

    def timing_per_frame_plot_stackedbar(self):
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

        FinalCutEval = list(self.times_final_evaluation_processing_full_frame.values())
        FinalCut = list(self.IO_EVAL_cut_crops_final.values())
        FinalEval = np.array(FinalCutEval) - np.array(FinalCut)

        postprocess = list(self.postprocess.values())

        IO_loads = np.array(IO_loads)
        IO_saves = np.array(IO_saves)
        if len(IO_loads) > len(IO_saves):
            IO_loads = IO_loads[0:len(IO_saves)]

        AttWait = np.array(AttWait)
        FinalCutEval = np.array(FinalCutEval)
        FinalCut = np.array(FinalCut)
        postprocess = np.array(postprocess)

        N = len(IO_loads)
        ind = np.arange(N)
        width = 0.35

        plt.title("Per frame analysis - stacked bar")
        plt.ylabel("Time (s)")
        plt.xlabel("Frame #num")

        p1 = plt.bar(ind, IO_loads, width, color='yellow') # yerr=stand deviation
        p2 = plt.bar(ind, IO_saves, width, bottom=IO_loads, color='orange')
        p3 = plt.bar(ind, AttWait, width, bottom=IO_saves+IO_loads, color='blue')
        #p4 = plt.bar(ind, FinalCutEval, width, bottom=AttWait+IO_saves+IO_loads, color='red')
        p4a = plt.bar(ind, FinalCut, width, bottom=AttWait+IO_saves+IO_loads, color='lightcoral')
        p4b = plt.bar(ind, FinalEval, width, bottom=FinalCut+AttWait+IO_saves+IO_loads, color='red')
        p5 = plt.bar(ind, postprocess, width, bottom=FinalEval+FinalCut+AttWait+IO_saves+IO_loads, color='green')

        if y_limit:
            ymin, ymax = plt.ylim()
            if ymax < 1.0:
                plt.ylim(0.0, 1.0)

        plt.legend((p1[0], p2[0], p3[0], p4a[0], p4b[0], p5[0]), ('IO loads', 'IO saves', attention_name, 'FinalCut', 'FinalEval', 'postprocess'))

        save_path = self.settings.render_folder_name + "stacked.png"
        plt.savefig(save_path, dpi=120)

        #plt.show()
        plt.clf()
        return 0

    def print_all_datalists(self):

        # prepare lists:
        """
        IO_loads = list(self.IO_loads.values())
        IO_saves = list(self.IO_saves.values())
        AttWait = list(self.times_attention_evaluation_waiting.values())
        AttEval = list(self.times_attention_evaluation_processing_full_frame.values())
        FinalCutEval = list(self.times_final_evaluation_processing_full_frame.values())
        FinalCut = list(self.IO_EVAL_cut_crops_final.values())
        postprocess = list(self.postprocess.values())

        print("IO_loads, IO_saves, AttWait/AttEval, FinalCutEval, postprocess")
        print("IO_loads",IO_loads)
        print("IO_saves",IO_saves)
        print("AttWait",AttWait)
        print("AttEval",AttEval)
        print("FinalCutEval",FinalCutEval)
        print("postprocess",postprocess)
        """
        print("IO_loads, IO_saves, AttWait/AttEval, FinalCutEval, postprocess")
        print("IO_loads",self.IO_loads)
        print("IO_saves",self.IO_saves)
        print("AttWait",self.times_attention_evaluation_waiting)
        print("AttEval",self.times_attention_evaluation_processing_full_frame)
        print("FinalCutEval",self.times_final_evaluation_processing_full_frame)
        print("FinalCut",self.IO_EVAL_cut_crops_final)
        print("postprocess",self.postprocess)


        return 0


    def save_whole_history_and_settings(self):
        settings = self.settings
        history = self
        folder = settings.render_folder_name

        settings.debugger = None
        save_object(settings, folder+"settings_last.pkl")
        save_object(history, folder+"history_last.pkl")

import pickle
def save_object(obj, filename):
    print(obj)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#def load_object(filename):
#    with open(filename, 'rb') as input:
#        obj = pickle.load(input)
#        return obj