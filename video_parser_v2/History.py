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
        Evaluation - can enter the evaluation time for attention + final evaluation stage;
                     this should also include how many crops were evaluated
                     (we are interested in performance per frame and also per crop)
        AttentionModel - can enter the number of active crops in each frame

        Renderer - can enter number of bounding boxes in final image

    """

    def __init__(self, settings):
        self.settings = settings
        self.frame_ticker = self.settings.render_history_every_k_frames

        # AttentionModel feed the number of active crops / all crops
        self.active_crops_per_frames = []
        self.total_crops_per_frames = []

        self.total_crops_during_attention_evaluation = []

        # Evaluation and VideoCapture feed time for
        self.times_attention_evaluation_processing_full_frame = []
        self.times_final_evaluation_processing_full_frame = []

        self.times_evaluation_each_loop = []

        # Renderer
        self.number_of_detected_objects = []

        # is it worth it to measure each crops evaluation time? maybe not really

        self.frame_number = -1

        self.loop_timer = None

    def tick_loop(self, force=False):
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

            self.times_evaluation_each_loop.append(t)

            self.end_of_frame(force)

    def report_crops_in_attention_evaluation(self, number_of_attention_crops):
        self.total_crops_during_attention_evaluation.append( number_of_attention_crops )

    def report_attention(self, active, total):
        self.active_crops_per_frames.append( active )
        self.total_crops_per_frames.append( total )

    def report_evaluation_whole_function(self, type, for_frame):
        if type == 'attention':
            self.times_attention_evaluation_processing_full_frame.append(for_frame)
        elif type == 'evaluation':
            self.times_final_evaluation_processing_full_frame.append(for_frame)

    def report_number_of_detected_objects(self, number_of_detected_objects):
        self.number_of_detected_objects.append(number_of_detected_objects)

    def report_skipped_final_evaluation(self):
        # No active crop was detected - we can just skip the finner evaluation
        self.times_final_evaluation_processing_full_frame.append(0)
        self.number_of_detected_objects.append(0)

    def end_of_frame(self, force=False):
        self.frame_ticker -= 1
        if self.frame_ticker <= 0 or force:

            print("History report!")
            self.frame_ticker = self.settings.render_history_every_k_frames

            self.plot_and_save()

    def plot_and_save(self):
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

        attention_plus_evaluation = [sum(x) for x in zip(self.times_final_evaluation_processing_full_frame, self.times_attention_evaluation_processing_full_frame)]

        if style == "areas":
            x = range(len(self.times_attention_evaluation_processing_full_frame))
            plt.fill_between(x, 0, self.times_attention_evaluation_processing_full_frame, label="Attention")
            plt.fill_between(x, self.times_attention_evaluation_processing_full_frame, attention_plus_evaluation, label="Attention+Final")
            plt.fill_between(x, attention_plus_evaluation, self.times_evaluation_each_loop, label="Whole loop") # 34 and 35
        elif style == "lines":
            plt.plot(self.times_attention_evaluation_processing_full_frame, label="Attention")
            plt.plot(attention_plus_evaluation, label="Attention+Final")
            plt.plot(self.times_evaluation_each_loop, label="Whole loop")

        self.limit_y_against_outliers(plt, self.times_evaluation_each_loop)

        plt.legend(loc='upper left', shadow=True)

        # active crops
        # - plot line for active
        # - plot line for total

        plt.subplot(212)
        if color_change: plt.gca().set_prop_cycle(cycler('color', color_list))

        self.prepare_plot(plt, "", "frames", "number of crops") # Active crops (includes attention)

        # ! this is number in final evaluation
        active_plus_attention = [sum(x) for x in zip(self.active_crops_per_frames,
                                                         self.total_crops_during_attention_evaluation)]
        total_plus_attention = [sum(x) for x in zip(self.total_crops_per_frames,
                                                         self.total_crops_during_attention_evaluation)]

        if style == "areas":
            x = range(len(self.total_crops_per_frames))
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