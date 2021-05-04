import logging
import inspect
import os
import functools
import time


class EndOfNameFilter(logging.Filter):
    def filter(self, record):
        record.trunc_name = record.name[-35:]
        return True


def get_logger(name=None, logging_format=None):
    if not logging_format:
        json_logging_format = """{"class_name": "%(name)s", "time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}"""
        logging_format = (
            "%(trunc_name)35.35s - %(asctime)-15s - %(levelname)5.5s: %(message)s"
        )
    if not name:
        name = os.path.basename(inspect.stack()[1].filename)
    logger = logging.getLogger(name)
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(logging_format))
        logger.addHandler(sh)
        sh.addFilter(EndOfNameFilter())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def logger_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if hasattr(func, "__self__"):
            if func.__self__.logger_level != "off":
                func.__self__.logger.debug(
                    "{func.__qualname__!s} called with arguments: {a}, and kwargs: {k}".format(
                        func=func,
                        a=", ".join([str(a) for a in args]),
                        k=", ".join(kwargs),
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


class test:
    def __init__(self, logger):
        self.logger = logger
        self.logger_level = "DEBUG"
        self.debug = True
        if self.debug:
            self.wrap_debug()

    def wrap_debug(self):
        for k in self.__dir__():
            v = getattr(self, k)
            if callable(v) and hasattr(v, "__self__"):  # only bound methods
                setattr(self, k, logger_wrapper(v))

    def poop(self, h):
        time.sleep(1)
        return h + 1
