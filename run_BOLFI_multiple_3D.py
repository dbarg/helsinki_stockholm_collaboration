# Run a BOLFI reconstructions

import sys
import pickle
import numpy as np
import scipy.stats as sps
from scipy.special import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

import elfi

from abc_reconstruction.model import Model
from abc_reconstruction.prior import BoundedNormal_x, BoundedNormal_y
from abc_reconstruction.utils import PriorPosition, minimize, ConstraintLCBSC

from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.utils import ModelPrior

plt.ioff()

class PoissonPrior(elfi.Distribution):
    '''Test workaround for discrete energy distribution'''
    def rvs(mu, size=1, random_state=None):
        return sps.poisson.rvs(mu=mu, loc=0, size=size, random_state=random_state)

    def pdf(k, mu):
        return mu**k * np.exp(-mu) / gamma(k + 1)

class BOLFIModel(object):
    def build(self, model, pattern,
              prior_pos, prior_cov = 25, r_bound = 47.9, pmt_mask = np.ones(127), pe=25):
        ### Build Priors
        pe = elfi.Prior(PoissonPrior, pe)  # TEST
        #pe = elfi.Prior('truncnorm', 0, 90, pe, pe**0.5)
        px = elfi.Prior(BoundedNormal_x, r_bound, prior_pos, prior_cov)
        py = elfi.Prior(BoundedNormal_y, px, r_bound, prior_pos, prior_cov)

        ### Build Model
        model=elfi.tools.vectorize(model)
        Y = elfi.Simulator(model, px, py, pe, observed=np.array([pattern]))

        def summarize(x, k):
            return np.array([e[k] for e in x])

        S1 = elfi.Summary(summarize, Y, 'energy')
        S2 = elfi.Summary(summarize, Y, 'time')

        de = elfi.Distance('braycurtis', S1)
        dt = elfi.Distance('braycurtis', S2)
        d = elfi.Operation(lambda a, b: a + b, de, dt)

        # TODO implement PMT mask here
        #d = elfi.Distance('braycurtis', Y)
        log_d = elfi.Operation(np.log, d)

        # set the ELFI model so we can remove it later
        self.model = px.model

        self.d0 = self.model.parameter_names.index('px')
        self.d1 = self.model.parameter_names.index('py')
        self.d2 = self.model.parameter_names.index('pe')

        ### Setup BOLFI
        bounds = {'px':(-r_bound, r_bound),
                  'py':(-r_bound, r_bound),
                  'pe':(0, 90)
                 }
        noise_vars = [5, 5, 5]
        noise_vars[self.d2] = 10

        target_model = GPyRegression(self.model.parameter_names,
                                     bounds=bounds)

        acquisition_method = ConstraintLCBSC(target_model,
                                             prior=ModelPrior(self.model),
                                             noise_var=noise_vars,
                                             exploration_rate=10)
        acquisition_method.d0 = self.d0
        acquisition_method.d1 = self.d1

        bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=50, update_interval=1,
                           # bounds=bounds,  # Not used when using target_model
                           target_model=target_model,
                           # acq_noise_var=[0.1, 0.1],  # Not used when using acq method
                           acquisition_method=acquisition_method,
                          )
        return bolfi

    def remove(self):
        # Clear the model from all nodes
        for node in ['px', 'py', 'pe', 'S1', 'S2', 'Y', 'de', 'dt', 'd', 'log_d']:
            self.model.remove_node(node)

class FM(object):
    def __init__(self, config_file):
        config_file_min = config_file[:-4] + "_minimal.ini"
        print("Initializing models from config files %s and %s" % (config_file,
                                                                   config_file_min))
        self.model = Model(config_file)
        self.min_model = Model(config_file_min)

    def change_defaults(self, timings=False, hitpattern='top', **kwargs):
        self.model.change_defaults(**kwargs)
        self.min_model.change_defaults(**kwargs)

        if timings:
            self.model.output_timing = True
            self.min_model.output_timing = True

        self.model.hitpattern(hitpattern)
        self.min_model.hitpattern(hitpattern)

    def get_models(self):
        return self.model, self.min_model

def run_BOLFI(truepos, start=0, stop=-1, folder='./'):
    ### Setup inits both the normal and minimal FM
    forward_models = FM('XENON1T_ABC_all_pmts_on.ini')
    # Change default simulation parameters and simulator modes
    forward_models.change_defaults(s2_electrons = 25,
                                   # z = -50,
                                   timings=True,
                                   hitpattern='full')

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
        #prior_pos = prior_mean(pattern['energy'])
        # The TPF position
        prior_pos = (pax_pos['PosRecTopPatternFit']['x'],
                     pax_pos['PosRecTopPatternFit']['y'])

        # The pax s2 raw energy estimate (at z=0, so no lifetime correction)
        s2 = model.output_plugin.last_event.main_s2
        e_cor = s2.s2_spatial_correction * s2.s2_saturation_correction

        # LY correction not needed, already in s2_spatial
        #ly_cor = model.pax.simulator.s2_light_yield_map.get_value(prior_pos[0],
        #                                                          prior_pos[1])
        s2_secondary_sc_gain = model.pax.simulator.config['s2_secondary_sc_gain']  #  21.3
        pax_e = s2.area * e_cor / s2_secondary_sc_gain
        print('Pax energy %.2f' % pax_e)
                     
        # Radial bound
        r_bound = 47.9
        # The PMT mask (currently not used)
        # pmt_mask = model.pmt_mask[:127].astype(int)

        # Build the ELFI model and BOLFI instance
        bolfi_model = BOLFIModel()
        bolfi = bolfi_model.build(min_model, pattern, prior_pos, pe=pax_e)

        # Run BOLFI
        try:
            post = bolfi.fit(n_evidence=300)
        except:
            bolfi_model.remove()
            continue

        # Save the discrepancy plot
        bolfi.plot_discrepancy()
        plt.savefig(folder + 'bolfi_disc_%d.png' % (index + start), dpi = 150)
        plt.close()

        ## Save the surface plot
        #bolfi.plot_state()
        #plt.xlim(-50, 50)
        #plt.ylim(-50, 50)
        #for ax in plt.gcf().axes:
        #        ax.add_artist(plt.Circle((0,0), 47.9, color='red', fill=False, linestyle='--'))
        #plt.savefig(folder + 'bolfi_surf_%d.png' % (index + start), dpi = 150)
        #plt.close()

        # Sample from the BOLFI fit
        try:
            result_BOLFI = bolfi.sample(1000, info_freq=1000)
        except:
            bolfi_model.remove()
            continue

        # Get the BOLFI output
        samples = result_BOLFI.samples_array
        means = result_BOLFI.sample_means
        modes = sps.mode(samples).mode[0]
        medians = np.median(samples, axis=0)

        pax_pos['truth'] = {'x': truth[0], 'y': truth[1]}
        pax_pos['BOLFI_mean'] = {'x': means['px'], 'y': means['py']}
        pax_pos['BOLFI_mode'] = {'x': modes[bolfi_model.d0], 'y': modes[bolfi_model.d1]}
        pax_pos['BOLFI_median'] = {'x': medians[bolfi_model.d0], 'y': medians[bolfi_model.d1]}
        pax_pos['BOLFI_mean_e'] = {'e': means['pe']}
        pax_pos['BOLFI_median_e'] = {'e': medians[bolfi_model.d2]}
        pax_pos['pax_e'] = pax_e

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

        true_pos = np.loadtxt('data/truepos_outer30.txt')
        #true_pos = np.loadtxt('data/truepos')

        run_BOLFI(true_pos, start=start, stop=stop, folder=folder)
    else:
        print("Usage: run_BOLFI_single.py start stop [output_folder]")
