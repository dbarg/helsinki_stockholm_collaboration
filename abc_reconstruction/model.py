import os
import inspect

from pax.core import Processor


class Model():
    """Implements the forward model for ABC project.
    
       The forward model used here uses the full pax waveformsimulation
       as input and the pax processing for output.
       
       Given a certain x,y position the model will first simulate an
       event using the waveformsimulator. Then reconstruct this event
       with the processor. It will then return the hit pattern of the
       main S2 of the event.
    """

    def __init__(self, config_filename = 'XENON1T_ABC_all_pmts_on.ini'):
        # Get path to modified pax plugin
        model_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        mod_dir = os.path.join(model_dir, '..', 'pax_mod')
        # Setup pax using a custom configuration 'XENON1T_ABC.ini'
        self.pax = Processor(config_paths = [config_filename],
                             config_dict = {'plugin_paths': [mod_dir]})
        # Get access to the input plugin WaveformSimulatorInput
        # derived from the WaveformSimulator class
        self.input_plugin = self.pax.get_plugin_by_name('WaveformSimulatorInput')
        # Use DummyOutput, it does not write to any file but
        # stores the event in its last_event variable
        self.output_plugin = self.pax.get_plugin_by_name('DummyOutput')
        
        # Dirty workaround for not using data derived spe gains if all PMTs on
        if config_filename.startswith('XENON1T_ABC_all_pmts_on'):
            gain_sigmas = [0.35 * g for g in self.pax.config['DEFAULT']['gains']]
            self.pax.config['DEFAULT']['gain_sigmas'] = gain_sigmas
            self.pax.config['gain_sigmas'] = gain_sigmas
            self.input_plugin.config['gain_sigmas'] = gain_sigmas
            self.input_plugin.simulator.config['gain_sigmas'] = gain_sigmas
            
    
    def change_defaults(self, z = 0.0, t = 10000,
                        recoil_type = 'NR', s1_photons=50, s2_electrons = 25):
        # Set new default options
        # str() converts are necassary to work with WaveformSimulator
        new_defaults = {'g4_id': -1,
                        'x': 0.0,
                        'y': 0.0,
                        'z': str(z),
                        't': str(t),
                        'recoil_type': recoil_type,
                        's1_photons': str(s1_photons),
                        's2_electrons': str(s2_electrons),
                        }
        self.input_plugin.default_instruction = new_defaults.copy()

    def __call__(self, x, y):
        """Returns a hitpattern of s2_electrons
           for given x, y interaction position.
        """
        # Set new x,y position
        self.input_plugin.set_instruction_for_next_event(x, y)
        # Run the waveformsimulator and pax processor
        self.pax.run(clean_shutdown=False)
        # Return the top hit pattern of the main S2 of the processed event
        return self.output_plugin.last_event.main_s2.area_per_channel[:127]
    
    def get_latest_pax_position(self):
        # Return the reconstructed positions by pax for the last
        # processed event
        pos_rec = {}
        select_algorithm = ['PosRecNeuralNet', 'PosRecTopPatternFit']
        for rp in self.output_plugin.last_event.main_s2.reconstructed_positions:
            if rp.algorithm in select_algorithm:
                pos_rec[rp.algorithm] = {'x': rp.x,
                                         'y': rp.y}
        return pos_rec
