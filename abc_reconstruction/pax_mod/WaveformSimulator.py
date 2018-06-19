from pax.plugins.io.WaveformSimulator import WaveformSimulator

# WaveformSimulatorInput class overwriting various input and output modes
# of the parent WaveformSimulator class so that it can be used in a pax
# instance which can be called for single events.
# Needed to build forward model for the ABC analysis.


class WaveformSimulatorInput(WaveformSimulator):
    """Provides interface to the waveformsimulator.
       Derives from WaveformSimulator, needs to overwrite
       store_true_peak and shutdown (no output needed),
       implements getting and setting instructions for
       the next event.
    """
    def startup(self):
        self.all_truth_peaks = []
        self.simulator = self.processor.simulator

        # The str convert is necessary unfortunately because the
        # original code reads from a csv file and works with str objects.
        self.default_instruction = {'g4_id': -1,  # placeholder
                                    'x': 0.0,  # cm
                                    'y': 0.0,  # cm
                                    'z': str(0.0),  # cm
                                    't': str(10000),  # ns 100mus
                                    'recoil_type': 'NR',  # ER or NR
                                    's1_photons': str(50),  # number of photons
                                    's2_electrons': str(25),  # number of electrons
                                   }
        self.number_of_events = 1

    def shutdown(self):
        """Empty function.
           Do not write truth peaks to file here.
        """
        return

    def store_true_peaks(self, peak_type, g4_id, t, x, y, z, photon_times, electron_times=(), peak_top_fraction=0):
        """Empty function.
           Do not store truth peaks.
        """
        return

    def get_instructions_for_next_event(self):
        yield [self.instruction]

    def set_instruction_for_next_event(self, x, y, z=None, s2_electrons=None):
        """recoil_type, x, y, z, t, s1_photons, s2_electrons, g4_id=-1
        """
        self.instruction = self.default_instruction.copy()
        self.instruction['x'] = x
        self.instruction['y'] = y
        if z is not None:
            self.instruction['z'] = str(z)
        if s2_electrons is not None:
            self.instruction['s2_electrons'] = str(s2_electrons)
