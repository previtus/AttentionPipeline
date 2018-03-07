
class Connection(object):
    """
    Holds connection to server(s), handles sending and receiving to and from the right one.
    """

    def __init__(self, settings):
        self.settings = settings

    def handshake(self):
        if self.settings.verbosity >= 2:
            print("Connection init, (here will be handshake with server... now local)")

