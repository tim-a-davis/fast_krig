import queue

import multiprocessing
import fast_krig as fk


def make_workers(worker=None, worker_queue=None, result_queue=None, myname=None):
    if worker is None:
        return
    while True:
        try:
            message = worker_queue.get()
            method_name = message.get("method", None)
            method = getattr(worker, method_name)
            args = message.get("args", [])
            kwargs = message.get("kwargs", {})
            name = message.get("name", None)
        except queue.Empty:
            message = "None"
            pass
        except AttributeError:
            pass
        else:
            if name and (name != myname):
                pass
            else:
                fk.config.logger.info(
                    f"{multiprocessing.current_process()} calling {method.__name__}"
                )
                results = method(*args, **kwargs)
                result_queue.put(results)
                result_queue.put({"name": myname})
                return
