import numpy as np
import os

filename_pulse_shape = 'utils/pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file


def compute_normalized_pulse_shape_area():

    time_steps, amplitudes = np.loadtxt(filename_pulse_shape, unpack=True, skiprows=1)
    amplitudes = amplitudes / min(amplitudes)
    delta_t = np.trapz(amplitudes, time_steps)

    return delta_t