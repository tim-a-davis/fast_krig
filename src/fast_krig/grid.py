from typing import Union
import numpy as np
from contextlib import contextmanager
import fast_krig as fk
from fast_krig.utils import WorkForce
import time


class GridConstructor:
    """This is the  Grid constructor class.  This class is meant to be used as an
    in constructor for another class.  Either the Grid class or the Krig class.
    The reason this is not meant to be used  as a standalone is because it only contains
    the methods required to fill up a grid given logs and meta data.

    Additional methods and subclasses of numpy will contain the methods to krig the grid.
    """

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
        """[summary]

        Args:
            logs (list): A list of the Logs that are used to compose this grid.  The Logs should have
                all the required meta data to place them within this grid.
            z_range (Union[tuple, list], optional): The range of z-values to create the grid for.
                Small z-ranges will make the grid smaller, and therefore faster to krig. If left empty,
                the grid will be auto-sized to match the data. Defaults to [].
            x_range (Union[tuple, list], optional): The range of x-values to create the grid for.
                Small x-ranges will make the grid smaller, and therefore faster to krig. If left empty,
                the grid will be auto-sized to match the data. Defaults to [].
            y_range (Union[tuple, list], optional): The range of y-values to create the grid for.
                Small y-ranges will make the grid smaller, and therefore faster to krig. If left empty,
                the grid will be auto-sized to match the data. Defaults to [].
            z_delta (float, optional): The size in the z-dimension of the grid cells. Defaults to 1.
            xy_delta (float, optional): The size in the xy-dimensions of the grid cells. Defaults to 100.
            auto_range (bool, optional): If True, then the range of the grid will be automatically determined.
                Defaults to True.
            stream ([type], optional): The stream on the logs to use to fill the grid. Defaults to None.
            n_samples (int, optional): The number of samples to take in each krig step. Defaults to 500.
        """
        self.logger = fk.config.logger.getChild(self.__class__.__name__)
        self.logs = logs
        self.xy_delta = xy_delta
        self.z_delta = z_delta
        self.stream = stream
        if auto_range:
            self._get_auto_xy_range()
        else:
            self.x_range = x_range
            self.y_range = y_range
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
        """This method will calculate the xy-range of the grid automatically given the data."""
        x_coords, y_coords = zip(*[(log.x_coord, log.y_coord) for log in self.logs])
        y_range = (np.max(y_coords) - np.min(y_coords)) * 1.2
        x_range = (np.max(x_coords) - np.min(x_coords)) * 1.2
        self.center = (np.mean(x_coords), np.mean(y_coords))
        self.y_range = (self.center[1] - y_range / 2, self.center[1] + y_range / 2)
        self.x_range = (self.center[0] - x_range / 2, self.center[0] + x_range / 2)

    def _get_auto_z_range(self):
        """This method will calculate the z-range of the grid automatically given the data.

        Returns:
            tuple: The (min, max) of the z range for the grid.
        """
        indices = np.stack([log.index for log in self.logs]).T
        return (np.min(indices) - 1, np.max(indices) + 1)

    def _make_grid(self):
        """Makes the empty grid given the ranges and cell sizes.
        Initializes a zero grid, then fills with NaNs.  This method
        creates a new attribute called grid.
        """
        self.z = np.arange(self.z_range[0], self.z_range[1], self.z_delta)
        self.x = np.arange(self.x_range[0], self.x_range[1], self.xy_delta)
        self.y = np.arange(self.y_range[0], self.y_range[1], self.xy_delta)
        self.grid = np.empty(tuple(map(len, [self.x, self.y, self.z])))
        self.grid[:] = np.nan

    def _get_filled_slice(self):
        """Gets the subset of the grid that is currently filled.

        Returns:
            tuple: x, y, z components of the filled cells.
        """
        return tuple(
            zip(
                *[
                    (np.tile(x, (len(z))), np.tile(y, (len(z))), z)
                    for x, y, z in zip(self.fill_x, self.fill_y, self.fill_z)
                ]
            )
        )

    def _get_filled_xyz(self):
        """Gets the subset of the grid that is currently filled, then returns array of coords.

        Returns:
            np.array: the coordinates of the filled cells in an array.
        """
        return np.vstack(
            [np.stack(list(zip(x, y, z))) for x, y, z in zip(*self._get_filled_slice())]
        )

    def _fill_grid(self):
        """Fills the grid with the log data after all other properties have been determined."""
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
    """The main Grid class, a subclass of np.ndarray.  This class contains all the logic to perform sampled kriging on a given grid.
    This class uses the grid constructor to generate a grid, then cast that object as a view into a numpy array.
    The resulting numpy array is then updated with the other pertinent metadata of the grid.  This is handled
    through the __new__ method, the __array_finalize__ method, the __reduce__ method, and the __setstate__ method.

    The __new__ method handles the initial creation of the grid, which is how the GridConstructor object can be cast
    as a view.

    The __array_finalize__ methodd is called any time the array is created, copied, or sliced into a new view.
    In order to maintain properties through copies and view casting, this method needs to be overridden.

    The __reduce__ method is responsible for pickling the array object.  By overriding this method, the extra
    attributes of the grid can be added to the pickled state, and carried to subprocesses.

    The __setstate__ method carries out th unpickling operation.  Overriding this method allows us to take
    advantage of our custom pickled state, and add back in our extra properties into the unpickled object.
    """

    def __new__(cls, *args, **kwargs):
        """Create the grid and subclass it into a numpy view.  Then add extra parameters.

        Returns:
            Object: The new subclassed grid object.
        """
        grid = GridConstructor(*args, **kwargs)  # Make the grid
        obj = np.asarray(grid.grid).view(cls)
        for k, v in grid.__dict__.items():
            if k != "grid":
                setattr(obj, k, v)
        return obj

    def __array_finalize__(self, obj):
        """Maintain object attributes accross copies and viewcasting of array.

        Args:
            obj (Object): The class object.
        """
        if obj is None:
            return
        try:
            for k, v in obj.__dict__.items():
                setattr(self, k, v)
        except AttributeError:
            pass

    def __reduce__(self):
        """Override of the __reduce__ method of numpy.  Call reduce, then add extra pickled state params.

        Returns:
            tuple: The pickled state of the object.
        """
        pickled_state = super(Grid, self).__reduce__()
        unneeded = ["logs"]
        needed_items = {k: v for k, v in self.__dict__.items() if k not in unneeded}
        new_state = pickled_state[2] + (needed_items,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Unpickle the pickled state of the sublass to add in extra params.

        Args:
            state (tuple): The pickled state from __reduce__ method.
        """
        self.__dict__.update(state[-1])
        super(Grid, self).__setstate__(state[0:-1])

    def update_fill(self, filled_coords: np.ndarray, empty_coos_coords: np.ndarray):
        """Update the attributed filled, empty_coos, filled_coos after each krig step.
        These attributes keep track of which parts of the grid have been filled, and which
        remain empty.

        Args:
            filled_coords (np.ndarray): The new filled coordinates from the latest krig sample.
            empty_coos_coords (np.ndarray): The empty coordinates to delete from the empty_coos array.
        """
        self.filled[filled_coords] = True
        self.empty_coos = np.delete(self.empty_coos, empty_coos_coords)
        self.filled_coos = np.append(self.filled_coos, filled_coords)

    @contextmanager
    def get_sample(self):
        """Context manager responsible for ensuring an atomic sample.
        Generate a sample of the data, then yield it back.  If the sample is not filled,
        then don't update the coordinates.

        Yields:
            np.ndarray, np.ndarray: The empty coordinates to fill, and sample data to
                generate the variogram.
        """
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
        """Perform kriging on one sample subset of the data.

        Returns:
            int: The number of cells that were filled on this krig sample.
        """
        with self.get_sample() as sample:
            ex, ey, ez = e = sample[0]  # get the empty coordinate sample
            fx, fy, fz = f = sample[1]  # get the filled coordinate sample
            filled_samples = self[fx, fy, fz]  # get the values of the filled data
            dists = self.calculate_self_distance(
                f.copy(), xy_delta=self.xy_delta, z_delta=self.z_delta
            )  # calculate the pairwise distance in the filled samples
            semivariance = self.MSE(
                filled_samples
            )  # Get the semivariance for the filled samples
            model = self.get_fitted_model(
                dists, semivariance
            )  # get the variogram model
            sample_dists = self.sample_distance(
                f.copy(), e.copy(), xy_delta=self.xy_delta, z_delta=self.z_delta
            )  # Get the pairwise distance of the filled and empty samples
            calculated_semivariance = np.linalg.inv(
                model(dists)
            )  # Calculated semivariance
            sample_semivariance = model(sample_dists)  # Get the sample semivariance
            weights = (
                np.dot(calculated_semivariance, np.expand_dims(sample_semivariance, 2))
                .squeeze()
                .T
            )  # Calculate the kriging weights
            new_vals = np.sum(
                filled_samples * weights, axis=1
            )  # Use the model to calculate new values
            self[ex, ey, ez] = new_vals
        return ex.size  # The number of empty cells that were filled

    def krig(self, *args, **kwargs):
        """Krig the grid in sample_size chunks until the entire grid is filled.

        Returns:
            Grid: The kriged grid
        """
        t1 = time.time()
        while np.isnan(self).any():
            sample_size = self.krig_sample()
        t2 = time.time()
        self.logger.info(f"Finished in {round(t2 - t1, 2)} seconds")
        return self

    def done(self, *args, **kwargs):
        """Check to see if the grid is filled or not.

        Returns:
            *args: Return whatever is passed.
        """
        if not np.isnan(self).any():
            return args

    def clean_grid(self):
        """Empty the grid and set all values to 0

        Returns:
            Grid: The Grid object, now with 0 values
        """
        self[:] = 0
        return self

    def get_fitted_model(self, dists, vals):
        """Pull the model, then call autofit given the params.

        Args:
            dists (np.ndarray): The distance matrix.
            vals (np.ndarray): The value matrix

        Returns:
            Model: The model object with fitted params
        """
        model = fk.config.model(fk.config.model_kwargs)
        model.autofit(dists.ravel(), vals.ravel())
        return model

    @staticmethod
    def sample_distance(filled, empty, xy_delta=1, z_delta=1, two_D=True) -> np.ndarray:
        """Calculate the distance between each filled value and every empty value.

        Args:
            filled (np.ndarray): The filled distance matrix
            empty (np.ndarray): The empty distance matrix.
            xy_delta (int, optional): The xy_delta of the grid. Defaults to 1.
            z_delta (int, optional): The z-delta of the grid. Defaults to 1.
            two_D (bool, optional): Whether or not to calulate 2D or 3D. Defaults to True.

        Returns:
            np.ndarray: [description]
        """
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
    """The multiprocessed kriging object.  This class will take care of multiprocessing workers,
    as well as assigning grids to workers, distributing tasks, and collecting aggregate data.  See
    documentation on workforce for a complete breakdown of how this is accomplished.

    Args:
        WorkForce (Object): The workforce class
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Krig class. Arguments and kwargs are passed to Workforce and Grid."""
        super().__init__(*args, **kwargs)
        self.aggregate_mean = (
            kwargs.get("worker").copy().clean_grid()
        )  # initialize the grid to 0s
        self.aggregate_variance = (
            kwargs.get("worker").copy().clean_grid()
        )  # initialize the grid to 0s
        self.workload = 0  # The number of grids that have yet to be completed
        self.grids_out = 1  # The number of grids that have been completed

    def __call__(self, num_grids):
        """The main method for this class.  Call krig_multi, then manage all outputs.

        Args:
            num_grids (int): The number of grids to krig.
        """
        self.krig_multi(num_grids)
        self.manage_outputs()

    def krig_single(self):
        """Krig one single grid.  Add the workload and call the krig method."""
        self.krig()
        self.workload += 1
        self.check_workload()

    def krig_multi(self, num_grids):
        """Krig num_grids number of grids.

        Args:
            num_grids (int): The number of grids to krig.
        """
        for _ in range(num_grids):
            self.krig_single()

    def manage_outputs(self):
        """Manager for the workers once Krig has been called.
        Constantly check to see if any workers need to be created, destroyed, and
        monitor the progress of the work.
        """
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
        """Calculate the incremental mean from the output grid and
        the current grid.

        Args:
            output (Grid): The completed grid object from one of the workers.
        """
        self.aggregate_mean *= (self.grids_out - 1) / self.grids_out
        self.aggregate_mean += output * (1 / self.grids_out)

    def incrimental_variance(self, output):
        """Calculate the incremental standard deviation from the output grid and
        the current grid.

        Args:
            output (Grid): The completed grid object from one of the workers.
        """
        self.aggregate_variance = (
            (self.grids_out - 2) / (self.grids_out - 1)
        ) * self.aggregate_variance + (
            (1 / self.grids_out) * np.square((output - self.aggregate_mean))
        )

    def read_grid(self, output):
        """This method is responsible for managing the aggregation and workload counts
        Each time a grid is read off the queue, it is calculated and the counters are updated.

        Args:
            output (Grid): The completed grid object from one of the workers.
        """
        self.workload -= 1
        self.incrimental_mean(output)
        self.grids_out += 1
        self.incrimental_variance(output)
        self.logger.info(f"Workload is: {self.workload}")
        self.logger.info(f"grids out is: {self.grids_out}")
        self.check_workload()

    def terminate_worker(self, name):
        """This method kills a worker with a specific name.

        Args:
            name (str): The name of the worker to terminate.
        """
        [worker.terminate() for worker in self.workers if worker.name == name]

    def cleanup(self):
        """This method will get rid of workers that are no longer running."""
        self.workers = [worker for worker in self.workers if worker.is_alive()]

    def check_workload(self):
        """This method checks to see if new workers need to be started given the workload"""
        if self.workload >= len(self.workers):
            [self._spawn() for _ in range(2)]
