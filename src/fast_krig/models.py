import numpy as np
from scipy.optimize import curve_fit
import functools
import fast_krig as fk


class Exponential:
    def __init__(self, krig_range: float=None, sill: float=None):
        self.range = krig_range
        self.sill = sill
    def autofit(self, dist, vals):
        if self.range and self.sill: return
        if not self.sill: self.sill = vals.mean()
        func = functools.partial(self.variogram, self.sill)
        popt, pcov = curve_fit(func, dist, vals, p0=(dist.mean()))
        if np.isinf(pcov.squeeze()): raise Exception("Bad auto fit")
        self.range = popt[0]
    @staticmethod
    def variogram(sill, dist, krig_range):
        return sill*(1 - np.exp(-dist/krig_range))
    def __call__(self, dist):
        return self.variogram(self.sill, dist, self.range)


class Gaussian:
    def __init__(self, krig_range: float=None, sill: float=None):
        self.range = krig_range
        self.sill = sill
    def autofit(self, dist, vals):
        if not self.sill: self.sill = vals.mean()
        func = functools.partial(self.variogram, self.sill)
        popt, pcov = curve_fit(func, dist, vals, p0=(dist.mean()))
        if np.isinf(pcov.squeeze()): raise Exception("Bad auto fit")
        self.range = popt[0]
    @staticmethod
    def variogram(sill, dist, krig_range):
        return sill*(1 - np.exp(-np.square(dist)/np.square(krig_range)))
    def __call__(self, dist):
        return self.variogram(self.sill, dist, self.range)


"""
class Linear:
    raise NotImplementedError()


class Spherical:
    raise NotImplementedError()
"""