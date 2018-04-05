from timeit import default_timer as timer
from threading import Thread


class ThreadHerd(object):
    """
    Pool of threads working on function, specify how many are there in reserve.
    Use till they are free, if there's none, wait (aka loop till there's one free one)
    ((tell that there could be waiting involved)).
    """

    def __init__(self, N):
        self.herd_size = N
        self.thread_pool = []

    def check_dead_threads(self):
        self.thread_pool = [t for t in self.thread_pool if t.is_alive()]

    def assign_job_CAREFULLY_CAN_STALL(self, fuction, arguments):
        if len(self.thread_pool) >= self.herd_size:
            self.check_dead_threads()
            if len(self.thread_pool) >= self.herd_size:
                print("ThreadHerd had full capacity (all",self.herd_size,") we need to wait.")

                start_waiting = timer()
                while len(self.thread_pool) >= self.herd_size:
                    self.check_dead_threads()
                have_waited = timer() - start_waiting
                print("Executing, we had to wait for ",have_waited)

        # assign new thread:
        t = Thread(target=fuction, args=arguments)
        #t.daemon = True
        t.start()

        self.thread_pool.append(t)

    def wait_till_all_are_done(self):
        for t in self.thread_pool:
            t.join()

"""
# Example usage
sheeps = ThreadHerd(16)
sheeps.assign_job(render_image,[image,path])
sheeps.assign_job(render_image,[image,path,store_result_here])
sheeps.wait_till_all_are_done()
"""