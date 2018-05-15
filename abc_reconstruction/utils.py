import numpy as np
import scipy.optimize

from pax.configuration import load_configuration
from pax.plugins.io.WaveformSimulator import uniform_circle_rv

from elfi.methods.bo.acquisition import LCBSC


class PriorPosition():
    """Implements the calculation of the mean of a prior
       given a pattern (either from data or from the forward model)
    """
    
    def __init__(self):
        # Get some settings from the XENON1T detector configuration
        config = load_configuration('XENON1T')
        
        # PMT positions
        pmt_config = config['DEFAULT']['pmts']
        
        # List of dicts {'x': , 'y'}, which the position
        self.positions = [pmt['position'] for pmt in pmt_config][:127]

    def __call__(self, pattern):
        assert len(pattern) == 127
        # The id of the PMT that sees most light
        max_pmt = np.argmax(pattern)
        
        # The position of that PMT
        pos = self.positions[max_pmt]
        
        return pos['x'], pos['y']

class Generator():
    """Generates test hitpatterns by drawing from LCE maps (forward model)"""
    
    def __init__(self, model):
        # Get some settings from the XENON1T detector configuration
        config = load_configuration('XENON1T')
        self.tpc_radius = config['DEFAULT']['tpc_radius']
        
        # Use the forward model also as generator for now
        self.model = model
    
    def __call__(self):
        return self.model(*uniform_circle_rv(self.tpc_radius))

def pol_to_cart(r, phi):
    return r * np.cos(phi), r * np.sin(phi)

def cart_to_pol(x, y):
    return np.sqrt(x**2 + y**2), np.arctan2(y,x)

# From elfi.methods.bo.utils
# Change to use SLSQP minimization with constraints
# Temporary solution
def minimize(fun,
             bounds,
             cons=None,
             method='L-BFGS-B',
             grad=None,
             prior=None,
             n_start_points=10,
             maxiter=1000,
             random_state=None):
    """Find the minimum of function 'fun'.
    Parameters
    ----------
    fun : callable
        Function to minimize.
    bounds : list of tuples
        Bounds for each parameter.
    cons : dict
        Constraints.
    method : string
        Minimization method.
    grad : callable
        Gradient of fun or None.
    prior : scipy-like distribution object
        Used for sampling initialization points. If None, samples uniformly.
    n_start_points : int, optional
        Number of initialization points.
    maxiter : int, optional
        Maximum number of iterations.
    random_state : np.random.RandomState, optional
        Used only if no elfi.Priors given.
    Returns
    -------
    tuple of the found coordinates of minimum and the corresponding value.
    """
    ndim = len(bounds)
    start_points = np.empty((n_start_points, ndim))

    if prior is None:
        # Sample initial points uniformly within bounds
        # TODO: combine with the the bo.acquisition.UniformAcquisition method?
        random_state = random_state or np.random
        for i in range(ndim):
            start_points[:, i] = random_state.uniform(*bounds[i], n_start_points)
    else:
        start_points = prior.rvs(n_start_points, random_state=random_state)
        if len(start_points.shape) == 1:
            # Add possibly missing dimension when ndim=1
            start_points = start_points[:, None]
        for i in range(ndim):
            start_points[:, i] = np.clip(start_points[:, i], *bounds[i])

    # Run the optimisation from each initialization point.
    locs = []
    vals = np.empty(n_start_points)
    for i in range(n_start_points):
        result = scipy.optimize.minimize(fun, start_points[i, :],
                                         method=method, jac=grad, bounds=bounds, constraints=cons)
        locs.append(result['x'])
        vals[i] = result['fun']

    # Return the optimal case.
    ind_min = np.argmin(vals)
    locs_out = locs[ind_min]
    for i in range(ndim):
        locs_out[i] = np.clip(locs_out[i], *bounds[i])

    return locs[ind_min], vals[ind_min]

class ConstraintLCBSC(LCBSC):
    """Derived from LCBSC acquisition method in ELFI.
       Overwriting acquire function to add options to
       minimizer to handle model constraint.
    """

    def acquire(self, n, t=None):
        """Return the next batch of acquisition points.
        Gaussian noise ~N(0, self.noise_var) is added to the acquired points.
        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int
            Current acq_batch_index (starting from 0).
        Returns
        -------
        x : np.ndarray
            The shape is (n, input_dim)
        """
        #logger.debug('Acquiring the next batch of %d values', n)

        # Optimize the current minimum
        def obj(x):
            return self.evaluate(x, t)

        def grad_obj(x):
            return self.evaluate_gradient(x, t)

        xhat, _ = minimize(
            obj,
            self.model.bounds,
            ({'type': 'ineq', 'fun': lambda x : 47.9**2 - (x[0]**2 + x[1]**2)}),
            'SLSQP',
            grad_obj,
            self.prior,
            self.n_inits,
            self.max_opt_iters,
            random_state=self.random_state)

        # Create n copies of the minimum
        x = np.tile(xhat, (n, 1))
        # Add noise for more efficient fitting of GP
        x = self._add_noise(x)

        return x
