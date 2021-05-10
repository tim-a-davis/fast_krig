import numpy as np
from scipy.optimize import curve_fit
import functools
import fast_krig as fk


class Exponential:
    """This class implements the Exponential semi-variogram model"""

    def __init__(self, krig_range: float = None, sill: float = None):
        """The initialization of the variogram model only requires a range and sill.

        Args:
            krig_range (float, optional): The max range of the model. Defaults to None.
            sill (float, optional): The max variance of the model. Defaults to None.
        """
        self.range = krig_range
        self.sill = sill

    def autofit(self, dist, vals):
        """Autofits the variogram parameters to the experimental data.

        Args:
            dist (np.ndarray): The distance matrix on  which to fit.
            vals (np.ndarray): The associated values for the distance array.

        Raises:
            Exception: [description]
        """
        if self.range and self.sill:
            return
        if not self.sill:
            self.sill = vals.mean()
        func = functools.partial(self.variogram, self.sill)
        popt, pcov = curve_fit(func, dist, vals, p0=(dist.mean()))
        if np.isinf(pcov.squeeze()):
            raise Exception("Bad auto fit")
        self.range = popt[0]

    @staticmethod
    def variogram(sill, dist, krig_range):
        """Static method to calculate exponential model

        Args:
            sill (float): The maximum variance of the model.
            dist (np.ndarray): The distance matrix.
            krig_range (float): The maximum range of the model.

        Returns:
            np.array: The exponential output from the model inputs.
        """
        return sill * (1 - np.exp(-dist / krig_range))

    def __call__(self, dist):
        return self.variogram(self.sill, dist, self.range)


class Gaussian:
    def __init__(self, krig_range: float = None, sill: float = None):
        """The initialization of this variogram model only requires a range and sill.

        Args:
            krig_range (float, optional): The max range of the model. Defaults to None.
            sill (float, optional): The max variance of the model. Defaults to None.
        """
        self.range = krig_range
        self.sill = sill

    def autofit(self, dist, vals):
        """Autofits the variogram parameters to the spherical data.

        Args:
            dist (np.ndarray): The distance matrix on  which to fit.
            vals (np.ndarray): The associated values for the distance array.

        Raises:
            Exception: [description]
        """
        if not self.sill:
            self.sill = vals.mean()
        func = functools.partial(self.variogram, self.sill)
        popt, pcov = curve_fit(func, dist, vals, p0=(dist.mean()))
        if np.isinf(pcov.squeeze()):
            raise Exception("Bad auto fit")
        self.range = popt[0]

    @staticmethod
    def variogram(sill, dist, krig_range):
        """Static method to calculate spherical model

        Args:
            sill (float): The maximum variance of the model.
            dist (np.ndarray): The distance matrix.
            krig_range (float): The maximum range of the model.

        Returns:
            np.array: The spherical output from the model inputs.
        """
        return sill * (1 - np.exp(-np.square(dist) / np.square(krig_range)))

    def __call__(self, dist):
        return self.variogram(self.sill, dist, self.range)
