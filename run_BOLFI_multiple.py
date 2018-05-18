# Run a BOLFI reconstructions

import sys
import pickle
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from functools import partial

import elfi

from abc_reconstruction.model import Model
from abc_reconstruction.prior import BoundedNormal_x, BoundedNormal_y
from abc_reconstruction.utils import PriorPosition, minimize, ConstraintLCBSC

from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.utils import ModelPrior


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

class BOLFIModel(object):
    def build(self, model, pattern,
              prior_pos, prior_cov = 64, r_bound = 47.8, pmt_mask = np.ones(127)):
        ### Build Priors
        px = elfi.Prior(BoundedNormal_x, r_bound, prior_pos, prior_cov)
        py = elfi.Prior(BoundedNormal_y, px, r_bound, prior_pos, prior_cov)

        ### Build Model
        model=elfi.tools.vectorize(model)
        Y = elfi.Simulator(model, px, py, observed=pattern)

        #likelihood_chisquare_masked = partial(likelihood_chisquare, w=pmt_mask)
        #log_d = elfi.Distance(likelihood_chisquare_masked, Y)

        #chisquare_masked = partial(chisquare, w=pmt_mask)
        #log_d = elfi.Distance(chisquare_masked, Y)

        #k2_test_masked = partial(k2_test, w=pmt_mask)
        #d = elfi.Distance(k2_test_masked, Y)
        #log_d = elfi.Operation(np.log, d)

        #sqrt_euclid_masked = partial(sqrt_euclid, w=pmt_mask)
        #d = elfi.Distance(sqrt_euclid_masked, Y)
        #log_d = elfi.Operation(np.log, d)

        d = elfi.Distance('euclidean', Y, w=pmt_mask)
        log_d = elfi.Operation(np.log, d)

        # set the ELFI model so we can remove it later
        self.model = px.model

        ### Setup BOLFI
        bounds = {'px':(-r_bound, r_bound), 'py':(-r_bound, r_bound)}

        target_model = GPyRegression(log_d.model.parameter_names,
                                     bounds=bounds)

        acquisition_method = ConstraintLCBSC(target_model,
                                             prior=ModelPrior(log_d.model),
                                             noise_var=[0.1, 0.1],
                                             exploration_rate=10)

        bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=1,
                           # bounds=bounds,  # Not used when using target_model
                           target_model=target_model,
                           # acq_noise_var=[0.1, 0.1],  # Not used when using acq method
                           acquisition_method=acquisition_method,
                          )
        return bolfi

    def remove(self):
        # Clear the model from all nodes
        self.model.remove_node('px')
        self.model.remove_node('py')
        self.model.remove_node('Y')
        self.model.remove_node('d')
        self.model.remove_node('log_d')

class FM(object):
    def __init__(self, config_file):
        config_file_min = config_file[:-4] + "_minimal.ini"
        print("Initializing models from config files %s and %s" % (config_file,
                                                                   config_file_min))
        self.model = Model(config_file)
        self.min_model = Model(config_file_min)

    def change_defaults(self, **kwargs):
        self.model.change_defaults(**kwargs)
        self.min_model.change_defaults(**kwargs)

    def get_models(self):
        return self.model, self.min_model

def run_BOLFI(truepos, start=0, stop=-1, folder='./'):
    ### Setup inits both the normal and minimal FM
    forward_models = FM('XENON1T_ABC_all_pmts_on.ini')
    forward_models.change_defaults(s2_electrons = 25)

    # Get the 'full' FM with complete pax reconstruction
    # so we also have the pax positions for comparison
    # Get the 'minimum' FM to use in the simulator
    model, min_model = forward_models.get_models()

    prior_mean = PriorPosition()

    for index, truth in enumerate(true_pos[start:stop]):
        print("Running BOLFI on index %d" % (index + start))

        # The pattern to reconstruct
        pattern = model(truth[0], truth[1])
        # The pax reconstruction of this pattern
        pax_pos = model.get_latest_pax_position()
        # The max PMT position
        prior_pos = prior_mean(pattern)

        # Radial bound (1mm less than TPC radius)
        r_bound = 47.8
        # The PMT mask
        pmt_mask = model.pmt_mask[:127].astype(int)

        # Build the ELFI model and BOLFI instance
        bolfi_model = BOLFIModel()
        bolfi = bolfi_model.build(min_model, pattern, prior_pos, pmt_mask=pmt_mask)

        # Run BOLFI
        try:
            post = bolfi.fit(n_evidence=200)
        except:
            bolfi_model.remove()
            continue

        # Save the discrepancy plot
        bolfi.plot_discrepancy()
        plt.savefig(folder + 'bolfi_disc_%d.png' % (index + start), dpi = 150)
        plt.close()

        # Save the surface plot
        bolfi.plot_state()
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        for ax in plt.gcf().axes:
                ax.add_artist(plt.Circle((0,0), 47.9, color='red', fill=False, linestyle='--'))
        plt.savefig(folder + 'bolfi_surf_%d.png' % (index + start), dpi = 150)
        plt.close()

        # Sample from the BOLFI fit
        result_BOLFI = bolfi.sample(1000, info_freq=1000)

        # Get the BOLFI output
        samples = result_BOLFI.samples_array
        means = result_BOLFI.sample_means
        modes = sps.mode(samples).mode[0]
        medians = np.median(samples, axis=0)

        pax_pos['truth'] = {'x': truth[0], 'y': truth[1]}
        pax_pos['BOLFI_mean'] = {'x': means['px'], 'y': means['py']}
        pax_pos['BOLFI_mode'] = {'x': modes[0], 'y': modes[1]}
        pax_pos['BOLFI_median'] = {'x': medians[0], 'y': medians[1]}

        # Save the output
        with open(folder + "bolfi_results_%d.pkl" % (index + start), 'wb') as f:
            pickle.dump(pax_pos, f)

        # Remove the model graph from memory so it can be reused in the next iteration
        bolfi_model.remove()

if __name__ == '__main__':
    if len(sys.argv) in [3, 4]:
        start = int(sys.argv[1])
        stop = int(sys.argv[2])
        if len(sys.argv) == 4:
            folder = sys.argv[3]
        else:
            folder = './'
        print('Running BOLFI indices %d through %d, storing results in %s' % (start,
                                                                              stop,
                                                                              folder))

        true_pos = np.loadtxt('data/truepos')

        run_BOLFI(true_pos, start=start, stop=stop, folder=folder)
    else:
        print("Usage: run_BOLFI_single.py start stop [output_folder]")
