from fast_krig.log import Log
import numpy as np


def generate_fake_log(start, end, delta, smin, smax, log=True, name="RESISTIVITY"):
    index = np.arange(start, end, delta).round(int(1 / delta))
    stream = np.random.random(len(index)) - 0.5
    stream = stream.cumsum()
    if log:
        stream = np.exp(stream)
    stream -= stream.min()
    stream = (smax - smin) * stream / (stream.max() - stream.min()) + smin
    arr = np.stack([index, stream]).T
    x_coord = np.random.random() * 30000 + 30000
    y_coord = np.random.random() * 30000 + 80000
    log = Log.from_numpy(
        arr, depth="DEPTH", labels=["DEPTH", name], x_coord=x_coord, y_coord=y_coord
    )
    return log
