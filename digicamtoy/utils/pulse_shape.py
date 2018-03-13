import numpy as np
import os
import scipy.interpolate
import os.path

__all__ = ['get_pulse_shape']


def compute_normalized_pulse_shape_area(linspace=(0, 100, 1000)):
    f = return_interpolant()
    time_steps = np.linspace(*linspace)
    amplitudes = f(time_steps)
    delta_t = np.trapz(amplitudes, time_steps)
    return delta_t


def compute_normalized_pulse_shape_area_square(linspace=(0, 100, 1000)):
    f = return_interpolant()
    time_steps = np.linspace(*linspace)
    amplitudes = f(time_steps)**2
    delta_t = np.trapz(amplitudes, time_steps)
    return delta_t


def compute_normalized_pulse_shape(
    pulse_shape_file_path=os.path.join(
        os.path.dirname(__file__),
        'pulse_SST-1M_AfterPreampLowGain.dat'
        )
):
    time_steps, amplitudes = np.loadtxt(
        pulse_shape_file_path,
        unpack=True
    )
    amplitudes /= amplitudes.max()
    return time_steps, amplitudes


def compute_mean_pulse_shape_value():
    time_steps, amplitudes = compute_normalized_pulse_shape()
    mean_time = np.average(time_steps, weights=amplitudes)
    f = return_interpolant()
    return f(mean_time)


def return_interpolant():
    time_steps, amplitudes = compute_normalized_pulse_shape()

    return scipy.interpolate.interp1d(
        time_steps,
        amplitudes,
        kind='cubic',
        bounds_error=False,
        fill_value=0.,
        assume_sorted=True
    )


def get_pulse_shape(
    time=np.arange(-10, 100, 4),
    t_0=0,
    amplitude=1,
    baseline=0,
    return_time=False
):
    f = return_interpolant()
    result = amplitude * f(time - t_0) + baseline
    if not return_time:
        return result
    else:
        return time, result


if __name__ == '__main__':

    print(
        'mean amplitude of pulse shape : %0.8f'
        % compute_mean_pulse_shape_value())
    print(
        'pulse shape area : %0.8f'
        % compute_normalized_pulse_shape_area())
    print(
        'pulse shape area square : %0.8f'
        % compute_normalized_pulse_shape_area_square())
