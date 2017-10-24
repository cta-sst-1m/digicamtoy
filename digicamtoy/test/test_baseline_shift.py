import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():

    def gain_drop(nsb_rate, cell_capacitance, bias_resistance):
        return 1. / (1. + nsb_rate * cell_capacitance * bias_resistance * 1E9)

    data_with_gain_drop = h5py.File('data/nsb_scan_1pixel_0_gain_drop_true.hdf5', 'r')
    data_without_gain_drop = h5py.File('data/nsb_scan_1pixel_0_gain_drop_false.hdf5', 'r')
    baseline_fadc = np.array(data_with_gain_drop['simulation_parameters']['baseline'])

    nsb = np.array(data_with_gain_drop['simulation_parameters']['nsb_rate'])
    n_nsb = len(nsb)
    baseline_shift = np.zeros((2,n_nsb))
    baseline_std = np.zeros((2,n_nsb))

    for dc_level in range(n_nsb):

        group_name = 'dc_level_%d_ac_level_0' % dc_level
        trace_with_gain_drop = np.array(data_with_gain_drop[group_name]['data'])
        trace_without_gain_drop = np.array(data_without_gain_drop[group_name]['data'])

        baseline_shift[0, dc_level] = np.mean(np.mean(trace_with_gain_drop - baseline_fadc[...,np.newaxis], axis=1), axis=1)
        baseline_std[0, dc_level] = np.mean(np.std(trace_with_gain_drop, axis=1), axis=1)
        baseline_shift[1, dc_level] = np.mean(np.mean(trace_without_gain_drop - baseline_fadc[...,np.newaxis], axis=1), axis=1)
        baseline_std[1, dc_level] = np.mean(np.std(trace_without_gain_drop, axis=1), axis=1)

    cell_capacitance = 85 * 1E-15
    bias_resistance = 1E4
    gain_drop = gain_drop(nsb/1E3, cell_capacitance, bias_resistance)

    plt.figure(figsize=(10, 10))
    plt.semilogx(nsb, baseline_shift[0], label='Gain drop : True')
    plt.semilogx(nsb, baseline_shift[1], label='Gain drop : False')
    plt.xlabel('$f_{NSB}$ [MHz]')
    plt.ylabel('baseline shift [LSB]')
    plt.legend()

    plt.figure(figsize=(10, 10))
    plt.semilogx(nsb, baseline_std[0], label='Gain drop : True')
    plt.semilogx(nsb, baseline_std[1], label='Gain drop : False')
    plt.xlabel('$f_{NSB}$ [MHz]')
    plt.ylabel('Standard deviation [LSB]')
    plt.legend()

    plt.figure(figsize=(10, 10))
    plt.semilogx(baseline_shift[0] / baseline_shift[1], baseline_std[0], label='Gain drop : True')
    plt.semilogx(baseline_shift[0] / baseline_shift[1], baseline_std[1], label='Gain drop : False')
    plt.xlabel('Gain drop []')
    plt.ylabel('Standard deviation [LSB]')
    plt.legend()

    plt.figure(figsize=(10, 10))
    plt.loglog(nsb, baseline_shift[0] / baseline_shift[1], label='Ratio')
    plt.loglog(nsb, gain_drop, label='Analytical', linestyle='--')
    plt.xlabel('$f_{NSB}$ [MHz]')
    plt.ylabel('Gain drop')
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':

    main()



