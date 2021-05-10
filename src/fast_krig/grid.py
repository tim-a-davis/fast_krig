from typing import Union
import numpy as np
from contextlib import contextmanager
import fast_krig as fk
from fast_krig.utils import WorkForce
import time


class GridConstructor:
    def __init__(
        self,
        logs: list,
        z_range: Union[tuple, list] = [],
        x_range: Union[tuple, list] = [],
        y_range: Union[tuple, list] = [],
        z_delta: float = 1,
        xy_delta: float = 100,
        auto_range: bool = True,
        stream=None,
        n_samples=500,
    ):
        self.logger = fk.config.logger.getChild(self.__class__.__name__)
        self.logs = logs
        self.xy_delta = xy_delta
        self.z_delta = z_delta
        self.stream = stream
        if auto_range:
            self._get_auto_xy_range()
        self.z_range = z_range or self._get_auto_z_range()
        self._make_grid()
        self._fill_grid()
        self.n_samples = n_samples
        self.sample_size = int(
            (self.grid.size - np.stack(self.fill_z).size) / self.n_samples
        )
        self.filled = ~np.isnan(self.grid).ravel()
        self.empty_coos = np.argwhere(~self.filled).T.squeeze()
        self.filled_coos = np.argwhere(self.filled).T.squeeze()
        self.coos = np.argwhere(self.grid)

    def _get_auto_xy_range(self):
        x_coords, y_coords = zip(*[(log.x_coord, log.y_coord) for log in self.logs])
        y_range = (np.max(y_coords) - np.min(y_coords)) * 1.2
        x_range = (np.max(x_coords) - np.min(x_coords)) * 1.2
        self.center = (np.mean(x_coords), np.mean(y_coords))
        self.y_range = (self.center[1] - y_range / 2, self.center[1] + y_range / 2)
        self.x_range = (self.center[0] - x_range / 2, self.center[0] + x_range / 2)

    def _get_auto_z_range(self):
        indices = np.stack([log.index for log in self.logs]).T
        return (np.min(indices) - 1, np.max(indices) + 1)

    def _make_grid(self):
        self.z = np.arange(self.z_range[0], self.z_range[1], self.z_delta)
        self.x = np.arange(self.x_range[0], self.x_range[1], self.xy_delta)
        self.y = np.arange(self.y_range[0], self.y_range[1], self.xy_delta)
        self.grid = np.empty(tuple(map(len, [self.x, self.y, self.z])))
        self.grid[:] = np.nan

    def _get_filled_slice(self):
        return tuple(
            zip(
                *[
                    (np.tile(x, (len(z))), np.tile(y, (len(z))), z)
                    for x, y, z in zip(self.fill_x, self.fill_y, self.fill_z)
                ]
            )
        )

    def _get_filled_xyz(self):
        return np.vstack(
            [np.stack(list(zip(x, y, z))) for x, y, z in zip(*self._get_filled_slice())]
        )

    def _fill_grid(self):
        self.fill_x, self.fill_y, self.fill_z, self.log_z = zip(
            *[
                (
                    np.abs(self.x - log.x_coord).argmin(),
                    np.abs(self.y - log.y_coord).argmin(),
                    np.where((self.z >= log.index.min()) & (self.z <= log.index.max()))[
                        0
                    ],
                    np.where((log.index >= self.z.min()) & (log.index <= self.z.max())),
                )
                for log in self.logs
            ]
        )
        self.grid[self._get_filled_slice()] = np.stack(
            [
                getattr(log, self.stream)[log_z]
                for log, log_z in zip(self.logs, self.log_z)
            ]
        )


class Grid(np.ndarray):
    def __new__(cls, *args, **kwargs):
        grid = GridConstructor(*args, **kwargs)
        obj = np.asarray(grid.grid).view(cls)
        for k, v in grid.__dict__.items():
            if k != "grid":
                setattr(obj, k, v)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        try:
            for k, v in obj.__dict__.items():
                setattr(self, k, v)
        except AttributeError:
            pass

    def __reduce__(self):
        pickled_state = super(Grid, self).__reduce__()
        unneeded = ["logs"]
        needed_items = {k: v for k, v in self.__dict__.items() if k not in unneeded}
        new_state = pickled_state[2] + (needed_items,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super(Grid, self).__setstate__(state[0:-1])

    def update_fill(self, filled_coords: np.ndarray, empty_coos_coords: np.ndarray):
        self.filled[filled_coords] = True
        self.empty_coos = np.delete(self.empty_coos, empty_coos_coords)
        self.filled_coos = np.append(self.filled_coos, filled_coords)

    @contextmanager
    def get_sample(self):
        max_sample = np.min([self.sample_size, self.empty_coos.size])
        empty_coos_coords = np.random.choice(
            np.arange(self.empty_coos.size), max_sample, replace=False
        )
        sample_points = self.empty_coos[empty_coos_coords]
        empty_coos = x, y, z = self.coos[sample_points].T
        filled_coos = self.coos[
            np.unique(np.random.choice(self.filled_coos, self.sample_size))
        ].T
        try:
            yield empty_coos, filled_coos
        finally:
            if not np.isnan(self[x, y, z]).any():
                self.update_fill(sample_points, empty_coos_coords)

    def krig_sample(self):
        with self.get_sample() as sample:
            ex, ey, ez = e = sample[0]
            fx, fy, fz = f = sample[1]
            filled_samples = self[fx, fy, fz]
            dists = self.calculate_self_distance(
                f.copy(), xy_delta=self.xy_delta, z_delta=self.z_delta
            )
            semivariance = self.MSE(filled_samples)
            model = self.get_fitted_model(dists, semivariance)
            sample_dists = self.sample_distance(
                f.copy(), e.copy(), xy_delta=self.xy_delta, z_delta=self.z_delta
            )
            calculated_semivariance = np.linalg.inv(model(dists))
            sample_semivariance = model(sample_dists)
            weights = (
                np.dot(calculated_semivariance, np.expand_dims(sample_semivariance, 2))
                .squeeze()
                .T
            )
            new_vals = np.sum(filled_samples * weights, axis=1)
            self[ex, ey, ez] = new_vals
        return ex.size  # The number of empty cells that were filled

    def krig(self, *args, sample_size=1, **kwargs):
        t1 = time.time()
        while np.isnan(self).any():
            sample_size = self.krig_sample()
        t2 = time.time()
        self.logger.info(f"Finished in {round(t2 - t1, 2)} seconds")
        return self

    def done(self, *args, **kwargs):
        if not np.isnan(self).any():
            return args

    def clean_grid(self):
        self[:] = 0
        return self

    def get_fitted_model(self, dists, vals):
        model = fk.config.model(fk.config.model_kwargs)
        model.autofit(dists.ravel(), vals.ravel())
        return model

    @staticmethod
    def sample_distance(filled, empty, xy_delta=1, z_delta=1, two_D=True) -> np.ndarray:
        filled[:2, :] *= xy_delta
        filled[2, :] *= z_delta
        empty[:2, :] *= xy_delta
        empty[2, :] *= z_delta
        if two_D:
            return np.sqrt(
                np.sum(
                    np.square(filled[:2, :] - np.expand_dims(empty[:2, :].T, 2)), axis=1
                )
            )
        else:
            return np.sqrt(
                np.sum(np.square(filled - np.expand_dims(empty.T, 2)), axis=1)
            )

    @staticmethod
    def calculate_self_distance(
        coords: np.ndarray, xy_delta=1, z_delta=1, two_D=True
    ) -> np.ndarray:
        """This method will calculate the distance from each point
        to every other point in the grid.  This can quickly lead to memory
        errors, so precautions should be taken to limit the size of the input
        coordinates.

        Args:
            coords (np.ndarray): The input coordinates for the filled datapoints.
                                 Should be of shape (3, no_datapoints)
            two_D (Bool): Whether or not to calculate a 2D distance instead of 3D.
                            Defaults to True.

        Returns:
            np.ndarray: An no_datapoints x no_datapoints grid of distances
        """
        coords[:2, :] *= xy_delta
        coords[2, :] *= z_delta
        if two_D:
            return np.sqrt(
                np.sum(
                    np.square(coords[:2, :] - np.expand_dims(coords[:2, :].T, 2)),
                    axis=1,
                )
            )
        else:
            return np.sqrt(
                np.sum(np.square(coords - np.expand_dims(coords.T, 2)), axis=1)
            )

    @staticmethod
    def MSE(vals: np.ndarray) -> np.ndarray:
        """This method will calculate the mean squared difference between every
        two values in the input array.  This is to generate values to go into the
        experimental variogram. This can quickly lead to memory errors, so
        precautions should be taken to limit the size of the input values.

        Args:
            vals (np.ndarray): The observed values in the sampled grid.

        Returns:
            np.ndarray: The squared distances of each point to every other point.
        """
        return (
            np.square(vals - np.expand_dims(vals.reshape(-1, 1), 1)).squeeze()
            / vals.size
        )


class Krig(WorkForce):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregate_mean = kwargs.get("worker").copy().clean_grid()
        self.aggregate_variance = kwargs.get("worker").copy().clean_grid()
        self.workload = 0
        self.grids_out = 1

    def __call__(self, num_grids):
        self.krig_multi(num_grids)
        self.manage_outputs()

    def krig_single(self):
        self.krig()
        self.workload += 1
        self.check_workload()

    def krig_multi(self, num_grids):
        for _ in range(num_grids):
            self.krig_single()

    def manage_outputs(self):
        self.logger.info("Waiting for outputs")
        while True:
            self.cleanup()
            self.check_workload()
            if self.workload == 0:
                break
            output = self._read()
            if isinstance(output, Grid):
                self.read_grid(output)
            elif isinstance(output, dict):
                self.terminate_worker(output.get("name", None))
                self.cleanup()

    def incrimental_mean(self, output):
        self.aggregate_mean *= (self.grids_out - 1) / self.grids_out
        self.aggregate_mean += output * (1 / self.grids_out)

    def incrimental_variance(self, output):
        self.aggregate_variance = (
            (self.grids_out - 2) / (self.grids_out - 1)
        ) * self.aggregate_variance + (
            (1 / self.grids_out) * np.square((output - self.aggregate_mean))
        )

    def read_grid(self, output):
        self.workload -= 1
        self.incrimental_mean(output)
        self.grids_out += 1
        self.incrimental_variance(output)
        self.logger.info(f"Workload is: {self.workload}")
        self.logger.info(f"grids out is: {self.grids_out}")
        self.check_workload()

    def terminate_worker(self, name):
        [worker.terminate() for worker in self.workers if worker.name == name]

    def cleanup(self):
        self.workers = [worker for worker in self.workers if worker.is_alive()]

    def check_workload(self):
        if self.workload >= len(self.workers):
            [self._spawn() for _ in range(2)]


# np.sqrt(np.sum(np.square(f - np.expand_dims(f.T, 2)), axis=1))

"""
from fast_krig.examples.logs import generate_fake_log
from fast_krig.grid import Grid
import numpy as np
import matplotlib.pyplot as plt

fake_logs = [
    generate_fake_log(9000, 10000, 1, 0.2, 5, log=False, name="RESISTIVITY")
    for i in range(3000)
]

self = Grid(fake_logs, stream="RESISTIVITY", z_range=(9900, 9901))
self2 = Grid(fake_logs, stream="RESISTIVITY", z_range=(9900, 9901))
self.krig()
self2.krig()
fig, ax = plt.subplots(1, 3, figsize=(16, 9))
fig.tight_layout()
ax[0].imshow(np.nan_to_num(self, 0))
ax[1].imshow(np.nan_to_num(self2, 0))
ax[2].imshow(np.nan_to_num(self2, 0) - np.nan_to_num(self, 0))
plt.show()




from fast_krig.examples.grids import generate_fake_grid
from fast_krig.grid import Grid
from fast_krig.utils import WorkForce
from fast_krig.grid import Krig
import fast_krig as fk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

grid = generate_fake_grid()

krig = Krig(worker=grid, max_workers=4)

krig(8)


krig(150)

fig, ax = plt.subplots(1, 2)
ax[0].set_title("Mean")
ax[1].set_title("Std Deviation")
for axs in ax:
    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)


sns.heatmap(np.array(krig.aggregate_variance).squeeze(), cmap="magma", ax=ax[1])
sns.heatmap(np.array(krig.aggregate_mean).squeeze(), cmap="magma", ax=ax[0])
plt.show()


plt.imshow(np.array(krig.aggregate_mean))
plt.show()





from fast_krig.examples.logs import generate_fake_log
from fast_krig.grid import Grid, GridConstructor
from fast_krig.utils import WorkForce
from fast_krig.grid import Krig
import fast_krig as fk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logs = [
    generate_fake_log(9000, 10000, 1, 0.2, 5, log=False, name="RESISTIVITY")
    for i in range(300)
]
grid = Grid(
    logs, 
    stream="RESISTIVITY", 
    z_range=(9900, 9901), 
    xy_delta=200
)

krig = Krig(worker=grid, max_workers=4)
krig(30)




images = []

sample_size = 1
while sample_size > 0:
    sample_size = grid.krig_sample()
    fig, ax = plt.subplots()
    h = sns.heatmap(np.nan_to_num(np.array(grid), 0).squeeze(), cmap="magma")
    sns.despine()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)
    plt.close("all")



imageio.mimsave('krig.gif', images, fps=15)


















from fast_krig import Dummy

import multiprocessing as mp
from fast_krig.utils import WorkForce

class Dummy:
    def print_args(*args):
        print(f"{mp.current_process().pid} is printing {args}")


workforce = WorkForce(worker=Dummy())
workforce._spawn()
workforce.print_args("Hello there!")





from fast_krig import WorkForce


workforce = WorkForce()
workforce.non_existent_method("Hello there", exclaim="!")

"""
