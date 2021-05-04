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
    x_coord = np.random.random() * 30000 + 30000
    y_coord = np.random.random() * 30000 + 80000
    dist_from_edge = np.sqrt(np.square(60000 - x_coord) + np.square(110000 - y_coord)) / 45000
    stream = stream * (1 - dist_from_edge)
    arr = np.stack([index, stream]).T
    log = Log.from_numpy(
        arr, depth="DEPTH", labels=["DEPTH", name], x_coord=x_coord, y_coord=y_coord
    )
    return log
