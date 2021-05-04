from typing import Union
import numpy as np
import fast_krig as fk


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
                    np.where((self.z >= log.index.min()) & (self.z <= log.index.max()))[0],
                    np.where((log.index >= self.z.min()) & (log.index <= self.z.max()))
                )
                for log in self.logs
            ]
        )
        self.grid[self._get_filled_slice()] = np.stack(
            [getattr(log, self.stream)[log_z] for log, log_z in zip(self.logs, self.log_z)]
        )


class Grid(np.ndarray):
    def __new__(cls, *args, **kwargs):
        grid = GridConstructor(*args, **kwargs)
        obj = np.asarray(grid.grid).view(cls)
        for k, v in grid.__dict__.items():
            if k != "grid":
                setattr(obj, k, v)
        return obj

    def get_sample(self, attr: str="empty_coos", sample_attr="sample_size"):
        return self.coos[
            np.unique(np.random.choice(getattr(self, attr), getattr(self, sample_attr)))
        ].T

    def krig_sample(self):
        e = ex, ey, ez = self.get_sample("empty_coos")
        f = fx, fy, fz = self.get_sample("filled_coos")
        filled_samples = self[fx, fy, fz]
        dists = self.calculate_self_distance(f.copy(), xy_delta=self.xy_delta, z_delta=self.z_delta)
        semivariance = self.MSE(filled_samples)
        model = self.get_fitted_model(dists, semivariance)
        sample_dists = self.sample_distance(f.copy(), e.copy(), xy_delta=self.xy_delta, z_delta=self.z_delta)
        calculated_semivariance = np.linalg.inv(model(dists))
        sample_semivariance = model(sample_dists)
        weights = np.dot(calculated_semivariance, np.expand_dims(sample_semivariance, 2)).squeeze().T
        new_vals = np.sum(filled_samples * weights, axis=1)
        self[ex, ey, ez] = new_vals
        
    
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
            return np.sqrt(np.sum(np.square(filled[:2, :] - np.expand_dims(empty[:2, :].T, 2)), axis=1))
        else:
            return np.sqrt(np.sum(np.square(filled - np.expand_dims(empty.T, 2)), axis=1))

    @staticmethod
    def calculate_self_distance(coords:np.ndarray, xy_delta=1, z_delta=1, two_D=True) -> np.ndarray:
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
            return np.sqrt(np.sum(np.square(coords[:2, :] - np.expand_dims(coords[:2, :].T, 2)), axis=1))
        else:
            return np.sqrt(np.sum(np.square(coords - np.expand_dims(coords.T, 2)), axis=1))
    
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
        return np.square(vals - np.expand_dims(vals.reshape(-1, 1), 1)).squeeze() / vals.size




#np.sqrt(np.sum(np.square(f - np.expand_dims(f.T, 2)), axis=1))

"""
from fast_krig.examples.logs import generate_fake_log
from fast_krig.grid import Grid
import numpy as np
import matplotlib.pyplot as plt

fake_logs = [
    generate_fake_log(9000, 10000, 1, 0.2, 2, log=False, name="RESISTIVITY")
    for i in range(3000)
]

self = Grid(fake_logs, stream="RESISTIVITY", z_range=(9900, 9901))


plt.imshow(np.nan_to_num(self, 0))
plt.show()

"""