from fast_krig.log import Log
import numpy as np


def generate_fake_log(start, end, delta, min, max, log=True):
    index = np.arange(start, end, delta).round(int(1/delta))
    stream = np.random.random(len(index)) - 0.5
    stream = stream.cumsum()
    if log:
        stream = np.exp(stream)
    stream -= stream.min()
    stream = (smax - smin) * stream / (stream.max() - stream.min()) + smin
    return np.stack([index, stream]).T
