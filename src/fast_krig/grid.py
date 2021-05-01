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
        if not z_range:
            self._get_auto_z_range()
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
        self.z_range = (np.min(indices) - 1, np.max(indices) + 1)

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
        self.fill_x, self.fill_y, self.fill_z = zip(
            *[
                (
                    np.abs(self.x - log.x_coord).argmin(),
                    np.abs(self.y - log.y_coord).argmin(),
                    np.where((self.z >= log.index.min()) & (self.z <= log.index.max()))[
                        0
                    ],
                )
                for log in self.logs
            ]
        )
        self.grid[self._get_filled_slice()] = np.stack(
            [getattr(log, self.stream) for log in self.logs]
        )


class Grid(np.ndarray):
    def __new__(cls, *args, **kwargs):
        grid = GridConstructor(*args, **kwargs)
        obj = np.asarray(grid.grid).view(cls)
        for k, v in grid.__dict__.items():
            if k != "grid":
                setattr(obj, k, v)
        return obj

    def get_sample(self, attr):
        return self.coos[
            np.unique(np.random.choice(getattr(self, attr), self.sample_size))
        ].T

    def krig_sample(self):
        ex, ey, ez = self.get_sample("empty_coos")
        fx, fy, fz = self.get_sample("filled_coos")
        filled_samples = self[fx, fy, fz]
