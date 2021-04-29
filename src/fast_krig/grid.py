import numpy as np
from fast_krig import krig_config


class Grid:
    def __init__(self):
        krig_config.logger_level = "WARNING"
        pass

    def print_it(self):
        return krig_config.show_options()

