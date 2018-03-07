from timeit import default_timer as timer

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

    """

    def __init__(self, settings):
        self.settings = settings

        self.active_crops_per_frames = []
        self.total_crops_per_frames = []

        self.times_evaluation_each_crop = []
        self.times_evaluation_each_frame = []
        self.times_evaluation_each_loop = []

        self.loop_timer = None

    def add_record(self, attention_evaluation, final_evaluation, active_coordinates):
        return 0

    def tick_loop(self):
        # measures every loop
        if self.loop_timer is None:
            self.loop_timer = timer()
        else:
            last = self.loop_timer
            self.loop_timer = timer()
            t = self.loop_timer - last
            print("Tick loop timer with t=", t)