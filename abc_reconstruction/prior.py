# Define custom ELFI priors

import numpy as np
import scipy.stats as sps

import elfi


class BoundedNormal_x(elfi.Distribution):
    def rvs(r_bound, mean, cov, size = 1, random_state = None):
        xs = np.zeros(size)
        for i in range(len(xs)):
            out_of_bounds = True
            while out_of_bounds:
                x,y = sps.multivariate_normal.rvs(mean = mean, cov = cov, size = 1, random_state = random_state)
                if x**2 + y**2 < r_bound**2:
                    out_of_bounds = False
                    xs[i] = x
        return xs

    def pdf(x, r_bound, mean=None, cov=1):
        # Take norm pdf, so wrong normalization
        return sps.norm.pdf(x, loc=mean[0], scale=np.sqrt(cov))

class BoundedNormal_y(elfi.Distribution):
    def rvs(x, r_bound, mean, cov, size = 1, random_state = None):
        max_ys = np.sqrt(r_bound**2 - x**2)
        std = np.sqrt(cov)
        a, b = (-max_ys - mean[1]) / std, (max_ys - mean[1]) / std
        return sps.truncnorm.rvs(a, b, loc = mean[1], scale = std, size = size, random_state =     random_state)

    def pdf(y, x, r_bound, mean=None, cov=1):
        # Take norm pdf, so wrong normalization
        vals = sps.norm.pdf(y, loc=mean[1], scale=np.sqrt(cov))
        # Check bounds
        out_of_bounds = x**2 + y**2 > r_bound**2
        vals[out_of_bounds] = 0
        return vals
