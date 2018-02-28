
class History(object):
    """
    Keeps interesting history such as which/how many crops were active, evaluation times, etc.
    Can also contain functions for plotting these into graphs.
    """

    def __init__(self, settings):
        self.settings = settings

    def add_record(self, attention_evaluation, final_evaluation, active_crops):
        return 0