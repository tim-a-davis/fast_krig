import queue

import multiprocessing
import fast_krig as fk


def make_workers(worker=None, worker_queue=None, result_queue=None, myname=None):
    """This function is meant to be passed to a multiprocessing worker.  The main
    idea is that the worker can be interacted with via a multiprocessing queue.  The
    main process sends messages to this queue.  This function reads the queue, and
    executes work with a worker object.  In this particular case, it is with a Grid object.

    The message should be in the following form:

        "method": The method on the worker to execute,
        "args": The arguments to the method,
        "kwargs": The keyword arguments to the method

    Args:
        worker (Object, optional): The worker object on which the methods are executed. Defaults to None.
        worker_queue (Queue, optional): The inlet multiprocessing queue where the worker reads the messages. Defaults to None.
        result_queue (Queue, optional): The outlet multiprocessing queue where the results are sent. Defaults to None.
        myname (str, optional): The name of the worker. Defaults to None.
    """    
    if worker is None:
        return
    while True:
        try:
            message = worker_queue.get() #get the message
            method_name = message.get("method", None) # extract the method name from the message
            method = getattr(worker, method_name) # get the method object from the worker
            args = message.get("args", []) # Get the args from the message
            kwargs = message.get("kwargs", {}) # get the kwargs from the message
            name = message.get("name", None) # get the name from the message
        except queue.Empty: # If the queue is empty, pass
            message = "None"
            pass
        except AttributeError:
            pass
        else:
            # If the name keyword is passed, that means that function can be called
            # on a specific. worker.  If name exists, it will only execute the method
            # if the names match.
            if name and (name != myname):
                pass # This should actually push the message back to the queue.
            else:
                fk.config.logger.info(
                    f"{multiprocessing.current_process()} calling {method.__name__}"
                )
                results = method(*args, **kwargs) # Execute the method
                result_queue.put(results) # Put the results in a queue
                result_queue.put({"name": myname}) # Put the name in the queue so that the 
                # orchestrator knows that this task has been completed.
                return
