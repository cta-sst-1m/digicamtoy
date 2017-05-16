import h5py
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    time_signal_10 = h5py.File('test/data/time_signal_0.hdf5', 'r')
    time_signal_50 = h5py.File('test/data/time_signal_1.hdf5', 'r')
    true_10 = time_signal_10['simulation_parameters']['time_signal'].value
    true_50 = time_signal_50['simulation_parameters']['time_signal'].value

    traces_10 = np.array(time_signal_10['dc_level_0_ac_level_0']['data'])
    traces_50 = np.array(time_signal_50['dc_level_0_ac_level_0']['data'])

    n_traces = traces_10.shape[-1]
    n_bins = traces_10.shape[1]
    x = np.arange(0, n_bins) * time_signal_10['simulation_parameters']['sampling_time']

    for i in range(3):

        plt.figure(figsize=(10,10))
        plt.step(x, traces_10[0, :, i], label=' signal at %0.2f ns' % true_10)
        plt.step(x, traces_50[0, :, i], label=' signal at %0.2f ns' % true_50)
        plt.xlabel('t [ns]')
        plt.ylabel('[LSB]')
        plt.legend()

    plt.show()


