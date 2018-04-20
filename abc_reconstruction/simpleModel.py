import numpy as np

from pax import utils
from pax.configuration import load_configuration
from pax.PatternFitter import PatternFitter


class SimpleModel():
    """Implements the simple forward model for ABC project.
    
       The forward model used here is the most basic test case.
       It draws from the per-PMT S2 LCE maps to provide a
       hitpattern for a given x,y. Assumes all top PMTs live.
       Also the total number of detected photo-electrons needs to
       be specified, this is set constant by default.
    """

    def __init__(self):
        # Get some settings from the XENON1T detector configuration
        config = load_configuration('XENON1T')
        
        # The per-PMT S2 LCE maps (and zoom factor which is a technical detail)
        lce_maps = config['WaveformSimulator']['s2_patterns_file']
        lce_map_zoom = config['WaveformSimulator']['s2_patterns_zoom_factor']

        # Simulate the right PMT response
        qes = np.array(config['DEFAULT']['quantum_efficiencies'])
        top_pmts = config['DEFAULT']['channels_top']
        errors = config['DEFAULT']['relative_qe_error'] + config['DEFAULT']['relative_gain_error']

        # Set up the PatternFitter which sample the LCE maps
        self.pf = PatternFitter(filename=utils.data_file_name(lce_maps),
                                zoom_factor=lce_map_zoom,
                                adjust_to_qe=qes[top_pmts],
                                default_errors=errors)

    def __call__(self, x, y, n_obs = 500):
        """Returns a hitpattern of n_obs photo-electrons
           for given x, y position.
        """
        
        return n_obs * self.pf.expected_pattern((x, y))
