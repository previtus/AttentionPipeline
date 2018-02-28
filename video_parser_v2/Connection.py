
class Connection(object):
    """
    Holds connection to server(s), handles sending and receiving to and from the right one.
    """

    def __init__(self, settings):
        self.settings = settings

    def handshake(self):
        print("Handshaking with server")
        print("[!!!] No server connected now, first faking it on local.")

