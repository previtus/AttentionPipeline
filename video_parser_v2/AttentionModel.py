
class AttentionModel(object):
    """
    Calculation of which crop should be active. Stands in the middle of two evaluations - attention evaluation
    should produce a rough estimation where we should look, this class determines which crops are active from it.
    """

    def __init__(self, settings):
        self.settings = settings

    def get_active_crops(self, evaluation):

        return 0