#global krig_config
from .._log import get_logger

class InvalidOption(Exception):
    pass

class Option:
    def __init__(self, default=None, valid=[]):
        self.val = default
        self.valid = valid
    def __set_name__(self, owner, name):
        self.name = name
    def __get__(self, obj, cls=None):
        if not obj:
            return self
        return self.val
    def __set__(self, obj, value):
        if self.valid and not (value in self.valid):
            raise InvalidOption(f"{value} if not a valid option for {self.name}")
        self.val = value


class Config:
    krigging_method = Option("gaussian", valid=["spherical", "linear", "exponential"])
    logger = Option(get_logger("Main"))

    def show_options(self):
        return {name: getattr(self, name) for name, val in self.__class__.__dict__.items() if isinstance(val, Option)}
