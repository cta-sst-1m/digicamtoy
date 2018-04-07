import numpy as np
import os
import scipy.interpolate
import os
dir = os.path.dirname(__file__)
filename_pulse_shape = dir + '/pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file

__all__ = ['get_pulse_shape']


def compute_normalized_pulse_shape_area():

    #time_steps, amplitudes = compute_normalized_pulse_shape()

    f = return_interpolant()
    time_steps = np.linspace(0, 100, 1000)
    amplitudes = np.zeros(time_steps.shape[0])

    for i in range(amplitudes.shape[0]):
        amplitudes[i] = f(time_steps[i])

    delta_t = np.trapz(amplitudes, time_steps)

    return delta_t


def compute_normalized_pulse_shape_area_square():

    #time_steps, amplitudes = compute_normalized_pulse_shape()

    f = return_interpolant()
    time_steps = np.linspace(0, 100, 1000)
    amplitudes = np.zeros(time_steps.shape[0])

    for i in range(amplitudes.shape[0]):
        amplitudes[i] = f(time_steps[i])**2

    delta_t = np.trapz(amplitudes, time_steps)

    return delta_t


def compute_normalized_pulse_shape():

    time_steps, amplitudes = np.loadtxt(filename_pulse_shape, unpack=True, skiprows=1)
    amplitudes = amplitudes / amplitudes.max()

    return time_steps, amplitudes


def compute_mean_pulse_shape_value():

    time_steps, amplitudes = compute_normalized_pulse_shape()
    mean_time = np.average(time_steps, weights=amplitudes)

    f = return_interpolant()
    return f(mean_time)


def return_interpolant():
    time_steps, amplitudes = compute_normalized_pulse_shape()

    return scipy.interpolate.interp1d(time_steps, amplitudes, kind='cubic', bounds_error=False, fill_value=0., assume_sorted=True)


def get_pulse_shape(time, t_0, amplitude, baseline=0):

    f = return_interpolant()

    return amplitude * f(time - t_0) + baseline


if __name__ == '__main__':

    print('mean amplitude of pulse shape : %0.8f' %compute_mean_pulse_shape_value())
    print('pulse shape area : %0.8f' %compute_normalized_pulse_shape_area())
    print('pulse shape area square : %0.8f' %compute_normalized_pulse_shape_area_square())
