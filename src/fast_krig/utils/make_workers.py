import queue


def make_workers(worker=None, worker_queue=None, result_queue=None):
    if not worker: return
    while True:
        try:
            message = worker_queue.get()
            method_name = message.get("method", None)
            method = getattr(worker, method_name)
            args = message.get("args", [])
            kwargs = message.get("kwargs", {})
        except queue.Empty:
            message = "None"
            pass
        except AttributeError:
            pass
        if message == "STOP":
            break
        elif message == "None":
            pass
        else:
            results = method(*args, **kwargs)
            result_queue.put(results)