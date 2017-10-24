from digicamtoy.core.tracegenerator import TraceGenerator
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import moment


def compute_moments_nsb(n_waveform, nsb_rate):

    digicam = [TraceGenerator(start_time=0., end_time=196., sampling_time=4., nsb_rate=nsb,
                              mean_crosstalk_production=0.08, gain_nsb_dependency=True, sigma_e=0.8, sigma_1=0.8,
                              gain=5.8, baseline=500.) for nsb in nsb_rate]

    std = np.zeros((len(digicam), n_waveform))
    waveform = np.zeros((len(digicam), n_waveform, (200) // digicam[0].sampling_time))

    for i in range(std.shape[0]):
        for j in range(std.shape[1]):
            digicam[i].__next__()
            waveform[i, j] = digicam[i].adc_count

    waveform = waveform.reshape((waveform.shape[0], waveform.shape[1] * waveform.shape[2]))
    n_bins = waveform.shape[1]
    mean = np.mean(waveform, axis=1)
    # moment_2 = moment(waveform, moment=2, axis=1)
    moment_4 = moment(waveform, moment=4, axis=1)
    std = np.std(waveform, axis=1)

    mean_error = std / np.sqrt(n_bins)
    moment_2_error = np.sqrt(1./n_bins * (moment_4 - (n_bins-3)/(n_bins-1) * std**4))
    std_error = 0.5 * moment_2_error / np.sqrt(std**2)
    # std_error = moment_2_error
    return mean, mean_error, std, std_error


if __name__ == '__main__':

    nsb_rate = np.logspace(0, np.log10(50000), num=30) * 1E6 * 1E-9
    mean_true, mean_error_true, std_true, std_error_true = compute_moments_nsb(n_waveform=1000, nsb_rate=nsb_rate)
    np.savez('true_mean_std_nsb.npz', mean=mean_true, mean_error=mean_error_true, std=std_true, std_error=std_error_true, nsb_rate=nsb_rate)
    data = np.load('true_mean_std_nsb.npz')
    mean_true = data['mean']
    mean_error_true = data['mean_error']
    std_true = data['std']
    std_error_true = data['std_error']
    nsb_rate = data['nsb_rate']

    print(nsb_rate)
    print(std_true)
    print(mean_true)

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.fill_between(nsb_rate * 1E3, mean_true - mean_error_true, mean_true + mean_error_true, alpha=0.3, color='k', label='true')
    axis.plot(nsb_rate * 1E3, mean_true, alpha=0.3, color='k')
    axis.set_xscale('log')
    axis.set_ylabel('Baseline [LSB]')
    axis.set_xlabel('$f_{nsb}$ [MHz]')
    axis.legend()

    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.fill_between(nsb_rate * 1E3, std_true - std_error_true, std_true + std_error_true, alpha=0.3, color='k', label='true')
    axis.plot(nsb_rate * 1E3, std_true, alpha=0.3, color='k')
    axis.set_xscale('log')
    axis.set_ylabel('STD [LSB]')
    axis.set_xlabel('$f_{nsb}$ [MHz]')
    axis.legend()

    plt.show()