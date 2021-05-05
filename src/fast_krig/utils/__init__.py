import functools
import fast_krig as fk
from multiprocessing import Process, Queue
import time
from .make_workers import make_workers


def make_worker(daemon=True):
    """ This is a decorator that puts the decorated function on a daemon thread """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            proc = Process(
                target=func, args=args, kwargs=kwargs, daemon=daemon
            )
            proc.start()
            return proc
        return wrapper
    return decorator


class WorkForce:
    def __init__(self, worker=None, inlet=Queue(), outlet=Queue(), daemon=True):
        self.worker = worker
        self.inlet = inlet
        self.outlet = outlet
        self.daemon = daemon
        self.workers = []
        self.logger = fk.config.logger.getChild(self.__class__.__name__)

    def __getattr__(self, attr):
        try:
            return super(WorkForce, self).__getattr__(attr)
        except AttributeError:
            return functools.partial(self.exec_method, method=attr)
    
    def _make_worker(self):
        func = make_worker(daemon=True)(make_workers)
        return func(worker=self.worker, worker_queue=self.inlet, result_queue=self.outlet)
    
    def _spawn(self):
        worker = self._make_worker()
        self.workers.append(worker)
        if worker.is_alive():
            self.logger.info(f"PID {worker.pid} is up and running")
        else:
            self.logger.info(f"PID {worker.pid} tried to start but died")
    
    def exec_method(self, *args, method=None, **kwargs):
        message = dict(
            method=method,
            args=args,
            kwargs=kwargs
        )
        self.inlet.put(message)


def make_workforce(worker=None, worker_queue=None, result_queue=None, daemon=True):
    """This function takes a class object (worker), and multiprocessing queues, and returns
    a partial method that, when executed, will return a process on which the class is running.

    The general idea is that you can create as many copies of this process as needed, perhaps in
    a list comprehension like so:

    >>> create_worker = make_workforce(worker=worker, worker_queue=q)
    >>> workers = [create_worker() for _ in range(1)]
    >>> workers
    [<Process name='Process-1' pid=79163 parent=79126 started daemon>]

    The general framework for interacting with the processes is to send a message with the 
    following format:

    {
        "method": The method on the worker to execute,
        "args": The arguments to the method,
        "kwargs": The keyword arguments to the method
    }

    Attributes:
        worker (Object): the worker object.
        worker_queue (Object): the queue object to distribute work to the threads.
        daemon (bool=True): whether or not to put the threads on a daemon

    Returns:
        A list of worker thread objects.
    """

    make_force = make_worker(daemon=True)(make_workers)
    return functools.partial(make_force, worker=worker, worker_queue=worker_queue, result_queue=result_queue)


def logger_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if hasattr(func, "__self__"):
            if func.__self__.logger.level <= 10:
                func.__self__.logger.debug(
                    "{func.__qualname__!s} called with arguments: {a}, and kwargs: {k}".format(
                        func=func,
                        a=", ".join([str(a) for a in args]),
                        k=", ".join([f"{k}: {v}" for k, v in kwargs.items()]),
                    )
                )
                time0 = time.time()
                func_out = func(*args, **kwargs)
                func.__self__.logger.debug("Function returned %s", func_out)
                func.__self__.logger.debug(
                    "Took %ss to execute", round(time.time() - time0, 3)
                )
                return func_out
            else:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapped


def wrap_debug(self):
    for k in self.__dir__():
        v = getattr(self, k)
        if callable(v) and hasattr(v, "__self__"):  # only bound methods
            setattr(self, k, logger_wrapper(v))

