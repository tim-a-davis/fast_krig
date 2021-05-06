from fast_krig.examples.logs import generate_fake_log
from fast_krig.grid import Grid


def generate_fake_grid(n_logs=5000):
    fake_logs = [
        generate_fake_log(9000, 10000, 1, 0.2, 5, log=False, name="RESISTIVITY")
        for i in range(5000)
    ]
    grid = Grid(fake_logs, stream="RESISTIVITY", z_range=(9900, 9901))
    return grid
