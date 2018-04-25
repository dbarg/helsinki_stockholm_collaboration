# Run a single BOLFI reconstruction

import sys
import pickle
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

import elfi

from abc_reconstruction.model import Model
from abc_reconstruction.prior import BoundedNormal_x, BoundedNormal_y
from abc_reconstruction.utils import PriorPosition


def run_BOLFI_single(index, true_x, true_y):
    ### Setup

    model = Model('XENON1T_ABC.ini')
    model.change_defaults(s2_electrons = 25)

    prior_mean = PriorPosition()

    pattern = model(true_x, true_y)
    pax_pos = model.get_latest_pax_position()
    prior_pos = prior_mean(pattern)

    r_bound = 47.9

    ### Build Priors
    px = elfi.Prior(BoundedNormal_x, r_bound, prior_pos, 64)
    py = elfi.Prior(BoundedNormal_y, px, r_bound, prior_pos, 64)

    ### Build Model
    model=elfi.tools.vectorize(model)
    Y = elfi.Simulator(model, px, py, observed=pattern)

    d = elfi.Distance('euclidean', Y)
    log_d = elfi.Operation(np.log, d)

    ### Setup BOLFI
    bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=1,
                       bounds={'px':(-r_bound, r_bound), 'py':(-r_bound, r_bound)},
                       acq_noise_var=[0.1, 0.1])

    ### Run BOLFI
    post = bolfi.fit(n_evidence=200)

    bolfi.plot_discrepancy()
    plt.savefig('bolfi_disc_%d.png' % index, dpi = 150)
    plt.close()

    result_BOLFI = bolfi.sample(1000, info_freq=1000)

    means = result_BOLFI.sample_means
    modes = sps.mode(result_BOLFI.samples_array).mode[0]

    pax_pos['truth'] = {'x': true_x, 'y': true_y}
    pax_pos['BOLFI_mean'] = {'x': means['px'], 'y': means['py']}
    pax_pos['BOLFI_mode'] = {'x': modes[0], 'y': modes[1]}
    return pax_pos

if __name__ == '__main__':
    i = int(sys.argv[1])
    print('Running BOLFI index', i)

    true_pos = np.loadtxt('data/truepos')

    result = run_BOLFI_single(i, true_pos[i][0], true_pos[i][1])

    with open("bolfi_result_%d.pkl" % i, 'wb') as f:
        pickle.dump(result, f)
