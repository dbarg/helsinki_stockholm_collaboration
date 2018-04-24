import numpy as np

from pax.configuration import load_configuration
from pax.plugins.io.WaveformSimulator import uniform_circle_rv


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
