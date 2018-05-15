# Run a single BOLFI reconstruction

import sys
import pickle
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from functools import partial

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
    pmt_mask = model.pmt_mask[:127].astype(int)

    ### Build Priors
    px = elfi.Prior(BoundedNormal_x, r_bound, prior_pos, 64)
    py = elfi.Prior(BoundedNormal_y, px, r_bound, prior_pos, 64)

    ### Build Model
    model=elfi.tools.vectorize(model)
    Y = elfi.Simulator(model, px, py, observed=pattern)


    def likelihood_chisquare(y, n, w=None):
        if w is not None:
            y = y[:,w.astype(bool)]
            n = n[:,w.astype(bool)]

        n = np.clip(n, 1e-10, None)
        y = np.clip(y, 1e-10, None)
        res = 2 * np.sum(y - n  + n * np.log(n/y), axis=1)
        lres = np.log(res)
        #if lres > 10:
        #    lres = np.ones(lres.shape) * 9
        return lres

    def chisquare(y, n, w=None):
        if w is not None:
            y = y[:,w.astype(bool)]
            n = n[:,w.astype(bool)]
        y = np.clip(y, 1e-1, None)
        #print('y shape', y.shape)
        #print('n shape', n.shape)
        chisq, p = sps.chisquare(n, y, axis=1)
        return np.array(np.log(chisq))

    def k2_test(y, n, w=None):
        if w is not None:
            y = y[:,w.astype(bool)]
            n = n[:,w.astype(bool)]

        #d, p = sps.ks_2samp(n, y)  # , axis=1)
        # ks_2samp does not have axis arg
        ds = [sps.ks_2samp(n[0], y[i])[0] for i in range(y.shape[0])]
        return np.array(ds)

    def sqrt_euclid(y, n, w=None):
        if w is not None:
            y = y[:,w.astype(bool)]
            n = n[:,w.astype(bool)]

        d = np.sum(np.sqrt(np.abs(y - n)), axis=1)
        return d

    likelihood_chisquare_masked = partial(likelihood_chisquare, w=pmt_mask)
    log_d = elfi.Distance(likelihood_chisquare_masked, Y)

    #chisquare_masked = partial(chisquare, w=pmt_mask)
    #log_d = elfi.Distance(chisquare_masked, Y)

    #k2_test_masked = partial(k2_test, w=pmt_mask)
    #d = elfi.Distance(k2_test_masked, Y)
    #log_d = elfi.Operation(np.log, d)

    #sqrt_euclid_masked = partial(sqrt_euclid, w=pmt_mask)
    #d = elfi.Distance(sqrt_euclid_masked, Y)
    #log_d = elfi.Operation(np.log, d)

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
    samples = result_BOLFI.samples_array

    means = result_BOLFI.sample_means
    modes = sps.mode(samples).mode[0]
    medians = np.median(samples, axis=0)

    pax_pos['truth'] = {'x': true_x, 'y': true_y}
    pax_pos['BOLFI_mean'] = {'x': means['px'], 'y': means['py']}
    pax_pos['BOLFI_mode'] = {'x': modes[0], 'y': modes[1]}
    pax_pos['BOLFI_median'] = {'x': medians[0], 'y': medians[1]}
    return pax_pos

if __name__ == '__main__':
    i = int(sys.argv[1])
    print('Running BOLFI index', i)

    true_pos = np.loadtxt('data/truepos')

    result = run_BOLFI_single(i, true_pos[i][0], true_pos[i][1])

    with open("bolfi_result_%d.pkl" % i, 'wb') as f:
        pickle.dump(result, f)
