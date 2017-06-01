import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils.pulse_shape import get_pulse_shape
from scipy.optimize import curve_fit

def main():


    time_signal_10 = h5py.File('test/data/time_signal_0.hdf5', 'r')
    time_signal_50 = h5py.File('test/data/time_signal_1.hdf5', 'r')
    true_10 = time_signal_10['simulation_parameters']['time_signal'].value
    true_50 = time_signal_50['simulation_parameters']['time_signal'].value

    traces_10 = np.array(time_signal_10['dc_level_0_ac_level_0']['data'])
    traces_50 = np.array(time_signal_50['dc_level_0_ac_level_0']['data'])

    n_traces = traces_10.shape[-1]
    n_bins = traces_10.shape[1]
    x = np.arange(0, n_bins) * time_signal_10['simulation_parameters']['sampling_time']
    x_fit = np.linspace(0, n_bins*4, 1000)

    for i in range(3):

        plt.figure(figsize=(10,10))
        plt.step(x, traces_10[0, :, i], label=' signal at %0.2f ns' % true_10, where='mid')
        plt.step(x, traces_50[0, :, i], label=' signal at %0.2f ns' % true_50, where='mid')
        p0 = [(np.argmax(traces_10[0, :, i]) - 4) * 4, np.max(traces_10[0, :, i] - 500), 500]
        popt, pcov = curve_fit(get_pulse_shape, x, traces_10[0, :, i], p0=p0)
        plt.plot(x_fit, get_pulse_shape(x_fit, popt[0], popt[1], popt[2]),
                 label=' pulse shape fitted : t_0 = %0.2f \n amplitude = %0.2f \n baseline = %0.2f' % (
                 popt[0], popt[1], popt[2]))

        p0 = [(np.argmax(traces_50[0, :, i]) - 4) * 4, np.max(traces_50[0, :, i] - 500), 500]
        popt, pcov = curve_fit(get_pulse_shape, x, traces_50[0, :, i], p0=p0)
        plt.plot(x_fit, get_pulse_shape(x_fit, popt[0], popt[1], popt[2]),
                 label=' pulse shape fitted : t_0 = %0.2f \n amplitude = %0.2f \n baseline = %0.2f' % (
                 popt[0], popt[1], popt[2]))

        plt.xlabel('t [ns]')
        plt.ylabel('[LSB]')
        plt.legend(fontsize=14, loc='best')


    #plt.figure(figsize=(10, 10))
    #plt.step(x, traces_10[0, :, 0], label=' signal at %0.2f ns' % true_10, where='mid')
    #plt.plot(x, get_pulse_shape(x, true_10, 5.6 * 10, 500), label=' pulse shape at %0.2f' % true_10)






    plt.show()

if __name__ == '__main__':
    main()



