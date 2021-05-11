import functools
import fast_krig as fk
from multiprocessing import Process, Queue
import queue
import time
from .make_workers import make_workers
import uuid


def make_worker(daemon=True, name=None):
    """ This is a decorator that puts the decorated function on a daemon process """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            proc = Process(
                target=func, args=args, kwargs=kwargs, daemon=daemon, name=name
            )
            proc.start()
            return proc

        return wrapper

    return decorator


class WorkForce:
    """This class is the main class for interacting with objects with multiprocessed workers.
    This class is responsible for spawning workers and intercepting method calls to send
    to the multiprocessing workers.

    The general idea is that you can simply create this object and pass in any other object
    as a worker like so:

    >>> workforce = WorkForce(worker=Grid)
    >>> workforce._spawn()
    Main.WorkForce - INFO: PID 8061 is up and running

    The general framework for interacting with the processes is to send a message with the
    following format:

        "method": The method on the worker to execute,
        "args": The arguments to the method,
        "kwargs": The keyword arguments to the method
    
    The multiprocessing workers have access to all the methods of the worker object passed.
    But the methods can be called on this worker like so:

    >>> workforce.krig()

    Even though `.krig()` is not a method defined in this class, method calls will be
    intercepted and sent to the workers.

    Attributes:
        worker (Object): the worker object.
        inlet (Queue): The inlet queue on which method calls are sent.
        outlet (Queue): The outlet queue on which results are sent.
        daemon (bool): Whether or not to put the workers on daemon.
        max_workers (int): The maximum number of workers to spawn. 

    Returns:
        A list of worker thread objects.
    """
    def __init__(
        self,
        *args,
        worker=None,
        inlet=Queue(),
        outlet=Queue(),
        daemon=True,
        max_workers=2,
        **kwargs,
    ):
        """Initialization for the workforce.

        Args:
            worker (Object, optional): The worker on which operations are performed. Defaults to None.
            inlet (Queue, optional): The inlet multiprocessing queue where the worker reads the messages. Defaults to Queue().
            outlet ([type], optional): The outlet multiprocessing queue where the results are sent. Defaults to Queue().
            daemon (bool, optional): Whether or not to put the workers on a daemon. Defaults to True.
            max_workers (int, optional): The number of workers. Defaults to 2.
        """    
        self.worker = worker
        self.inlet = inlet
        self.outlet = outlet
        self.daemon = daemon
        self.workers = []
        self.max_workers = max_workers
        self.logger = fk.config.logger.getChild(self.__class__.__name__)

    def __getattr__(self, attr):
        """Interception for method calls.
        If the method call on this class raises a key error, then the method is 
        routed to `_exec_method` to be sent to a worker.

        Args:
            attr (str): The method call to send to the worker.

        Returns:
            Callable: A partial execution of `_exec_method` with the method defined.
        """        
        try:
            return super(WorkForce, self).__getattr__(attr)
        except AttributeError:
            return functools.partial(self._exec_method, method=attr)

    def _make_worker(self):
        """Spin up a worker and add it to the list of workers.

        Returns:
            multiprocess: The multiprocess worker.
        """        
        name = str(uuid.uuid4())
        func = make_worker(daemon=True, name=name)(make_workers)
        return func(
            worker=self.worker,
            worker_queue=self.inlet,
            result_queue=self.outlet,
            myname=name,
        )

    def _spawn(self):
        """If there are not enough workers to fulfill the work, then a worker is
        spawned, as long as the max_workers is not reached.
        """        
        if len(self.workers) < self.max_workers:
            worker = self._make_worker()
            self.workers.append(worker)
            if worker.is_alive():
                self.logger.info(f"PID {worker.pid} is up and running")
            else:
                self.logger.info(f"PID {worker.pid} tried to start but died")
        else:
            pass

    def _exec_method(self, *args, method=None, **kwargs):
        """Make a message to send to the multiprocessing worker.

        Args:
            method (str, optional): The name of the method to call. Defaults to None.
        """        
        message = dict(method=method, args=args, kwargs=kwargs)
        self.inlet.put(message)

    def _read(self):
        """Read the outlet queue.

        Returns:
            Object: The outlet object.
        """        
        try:
            return self.outlet.get_nowait()
        except queue.Empty:
            pass


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
