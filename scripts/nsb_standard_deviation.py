from digicamtoy.core.tracegenerator import TraceGenerator
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    nsb_rate = np.logspace(0, 3.5, num=10) * 1E6 * 1E-9

    digicam = [TraceGenerator(start_time=0., end_time=200., sampling_time=4., nsb_rate=nsb,
                 mean_crosstalk_production=0.08, gain_nsb_dependency=True, sigma_e=0.8, sigma_1=0.8, gain=5.8, baseline=500.) for nsb in nsb_rate]

    n_waveform = 1000

    std = np.zeros((len(digicam), n_waveform))

    for i in range(std.shape[0]):
        for j in range(std.shape[1]):

            digicam[i].__next__()

            waveform = digicam[i].adc_count
            std[i, j] = np.std(waveform)

    mean_std = np.mean(std, axis=1)
    mean_std_error = np.std(std) / np.sqrt(std.shape[1])

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.errorbar(nsb_rate * 1E3, mean_std, yerr=mean_std_error, linestyle='None', marker='o')
    axis.set_xscale('log')
    axis.set_xlabel('$f_{nsb}$ [MHz]')
    axis.set_ylabel('STD [LSB]')

    plt.show()
