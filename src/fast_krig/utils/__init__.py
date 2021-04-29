import functools
import threading
import queue
import time


def spawn(num_workers=1):
    """ This is a decorator that makes a list of the outputs of the decorated function """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            workers = []
            for _ in range(num_workers):
                workers.append(func(*args, **kwargs))
            return workers

        return wrapper

    return decorator


def make_worker(daemon=True):
    """ This is a decorator that puts the decorated function on a daemon thread """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread = threading.Thread(
                target=func, args=args, kwargs=kwargs, daemon=daemon
            )
            thread.start()
            return thread

        return wrapper

    return decorator


def make_workforce(n_workers=1, worker=None, worker_queue=None, daemon=True):
    """This function takes a class with a .work() method and creates (n_worker) number of threads with that worker.
    If the "STOP" message is passed to the queue, it will shut down one worker.
    If the message is a dictionary, it will be unpacked as keyword arguments to the .work() method of the worker.

    Attributes:
        n_workers (int): the number of workers to spawn.
        worker (Object): the worker object.
        worker_queue (Object): the queue object to distribute work to the threads.
        daemon (bool=True): whether or not to put the threads on a daemon

    Returns:
        A list of worker thread objects.
    """

    @spawn(n_workers)
    @make_worker(daemon=True)
    def make_workers(*args, worker=None, worker_queue=None):
        if worker:
            while True:
                try:
                    message = worker_queue.get()
                except queue.Empty:
                    message = "None"
                    pass
                if message == "STOP":
                    break
                elif message == "None":
                    pass
                else:
                    worker.work(**message)

    return make_workers(worker=worker, worker_queue=worker_queue)


def logger_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if hasattr(func, "__self__"):
            if func.__self__.logger_level != "off":
                func.__self__.logger.debug(
                    "{func.__qualname__!s} called with arguments: {a}, and kwargs: {k}".format(
                        func=func, a=", ".join([str(a) for a in args]), k=", ".join(kwargs)
                    )
                )
                time0 = time.time()
                func_out = func(*args, **kwargs)
                func.__self__.logger.debug("Function returned %s", func_out)
                func.__self__.logger.debug("Took %ss to execute", round(time.time() - time0, 3))
                return func_out
            else:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapped


def wrap_debug(self):
    for k in self.__dir__():
        v = getattr(self, k)
        if callable(v) and hasattr(v, "__self__"): #only bound methods
            setattr(self, k, logger_wrapper(v))