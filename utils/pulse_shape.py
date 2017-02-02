import numpy as np
import os
import scipy.interpolate

filename_pulse_shape = 'utils/pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file



def compute_normalized_pulse_shape_area():

    delta_t = np.trapz(amplitudes, time_steps)

    return delta_t

def compute_normalized_pulse_shape():

    time_steps, amplitudes = np.loadtxt(filename_pulse_shape, unpack=True, skiprows=1)
    amplitudes = amplitudes / min(amplitudes)

    return time_steps, amplitudes

def return_interpolant():
    time_steps, amplitudes = compute_normalized_pulse_shape()

    return scipy.interpolate.interp1d(time_steps, amplitudes, kind='cubic', bounds_error=False, fill_value=0.)
