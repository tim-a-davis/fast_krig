import os
import numpy as np
import lasio
from pathlib import Path
import fast_krig as fk
from fast_krig.utils import wrap_debug
from typing import Union


class Log:
    def __init__(
        self,
        index: np.ndarray,
        streams: dict,
        x_coord: float = None,
        y_coord: float = None,
    ):
        self.index = index
        self.streams = streams
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.logger = fk.config.logger.getChild(self.__class__.__name__)
        for name, stream in self.streams.items():
            setattr(self, name, stream)
        wrap_debug(self)

    @classmethod
    def from_numpy(
        cls,
        log_arr: np.ndarray,
        depth: str = None,
        labels: list = None,
        x_coord: float = None,
        y_coord: float = None,
    ):
        if not depth:
            raise Exception("You need to pass the name of the depth array")
        assert len(log_arr.shape) == 2
        if len(labels) != log_arr.shape[-1]:

            raise Exception("The labels and array must have the same dimension")
        index = log_arr[:, labels.index(depth)]
        log_arr = np.delete(log_arr, labels.index(depth), axis=1)
        streams = {
            label: arr
            for label, arr in zip([l for l in labels if l != depth], log_arr.T)
        }
        return cls(index, streams, x_coord, y_coord)

    @classmethod
    def from_las(cls, path: Union[str, os.PathLike, Path]):
        raise NotImplementedError


"""
from fast_krig.log import Log
import fast_krig as fk
import numpy as np
a = np.arange(500)
b = np.random.random((500, 2))
arr = np.hstack([a.reshape(-1, 1), b])
depth = "DPT"
labels = ["DPT", "A", "B"]
x_coord = 5000
y_coord = 14450

resistivity = Log.from_numpy(arr, depth, labels, x_coord=x_coord, y_coord=y_coord)
resistivity.printt("45", with_unit="mpg")
fk.config.logger.setLevel(10)
resistivity.printt("45", with_unit="mpg")



import lasio


path = "data/sample_logs/49-005-30258.las"
log = lasio.read(path)

"""
