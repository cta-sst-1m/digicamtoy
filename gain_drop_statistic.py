import numpy as np
from scipy.interpolate import interp1d, splrep, splev, splder
import matplotlib.pyplot as plt
from digicamtoy.utils.analytical import gain_drop
from nsb_baseline import compute_moments_nsb


if __name__ == '__main__':

    n_waveforms = np.linspace(4, 1000, num=5)
    true_values = np.load('true_mean_std_nsb.npz')
    mean_true = true_values['mean']
    std_true = true_values['std']
    nsb_rate = true_values['nsb_rate']

    trials = 400
    mean = np.zeros((n_waveforms.shape[0], trials, len(nsb_rate)))
    std = np.zeros((n_waveforms.shape[0], trials, len(nsb_rate)))

    for i, n_waveform in enumerate(n_waveforms):

        for j in range(trials):

            print(i, j)
            out = compute_moments_nsb(n_waveform=n_waveform, nsb_rate=nsb_rate)
            mean[i, j] = out[0]
            std[i, j] = out[2]

    mean_mean = np.mean(mean, axis=1)
    mean_error = np.std(mean, axis=1)
    std_mean = np.mean(std, axis=1)
    std_error = np.std(std, axis=1)

    np.savez('measured_mean_std_nsb.npz',  mean=mean_mean, mean_error=mean_error, std=std_mean, std_error=std_error,
             nsb_rate=nsb_rate, n_waveforms=n_waveforms, trials=trials)

    data = np.load('measured_mean_std_nsb.npz')
    mean_mean = data['mean']
    mean_error = data['mean_error']
    std_mean = data['std']
    std_error = data['std_error']
    nsb_rate = data['nsb_rate']
    n_waveforms = data['n_waveforms']
    trials = data['trials']

    # std_to_gain_drop = interp1d(std_true, gain_drop(nsb_rate), fill_value='extrapolate')
    spline = splrep(std_true, gain_drop(nsb_rate))
    std_to_gain_drop = lambda x: splev(x, spline)
    std_error_to_gain_drop_error = lambda x: np.abs(splev(x, splder(spline, n=1)))

    fig = plt.figure()
    axis = fig.add_subplot(111)
    fig_1 = plt.figure()
    axis_1 = fig_1.add_subplot(111)

    for i, n_waveform in enumerate(n_waveforms):

        axis.fill_between(nsb_rate * 1E3, mean_mean[i] - mean_error[i], mean_mean[i] + mean_error[i], alpha=0.3,
                          label='$N_{bins} =$ %d' % (n_waveform * 50))
        axis.plot(nsb_rate * 1E3, mean_mean[i], alpha=0.3)
        axis_1.fill_between(nsb_rate * 1E3, std_mean[i] - std_error[i], std_mean[i] + std_error[i], alpha=0.3,
                            label='$N_{bins} =$ %d' % (n_waveform * 50))
        axis_1.plot(nsb_rate * 1E3, std_mean[i], alpha=0.3)

        """
        for j, nsb in enumerate(nsb_rate):

            fig_2 = plt.figure()
            axis_2 = fig_2.add_subplot(111)
            fig_3 = plt.figure()
            axis_3 = fig_3.add_subplot(111)
            axis_2.hist(mean[i, :, j], bins='auto', label='$N_{bins} =$ %d, $f_{nsb} = $ %0.2f' % (n_waveform * 50, nsb * 1E3))
            axis_3.hist(std[i, :, j], bins='auto', label='$N_{bins} =$ %d, $f_{nsb} = $ %0.2f' % (n_waveform * 50, nsb * 1E3))
            axis_2.set_xlabel('baseline - true [LSB}')
            axis_3.set_xlabel('STD - true [LSB}')
            axis_2.legend()
            axis_3.legend()
        """

    axis.set_xscale('log')
    axis.set_xlabel('$f_{nsb}$ [MHz]')
    axis.set_ylabel('baseline [LSB]')
    axis.legend()
    axis_1.set_xscale('log')
    axis_1.set_xlabel('$f_{nsb}$ [MHz]')
    axis_1.set_ylabel('STD [LSB]')
    axis_1.legend()

    fig = plt.figure()
    axis = fig.add_subplot(111)

    fig_1 = plt.figure()
    axis_1 = fig_1.add_subplot(111)

    for i, n_waveform in enumerate(n_waveforms):
        axis.fill_between(std_mean[i], std_to_gain_drop((std_mean - std_error)[i]),
                          std_to_gain_drop((std_mean + std_error)[i]), alpha=0.3,
                          label='$N_{bins} =$ %d' % (n_waveform * 50), interpolate=True)
        axis.plot(std_mean[i], std_to_gain_drop(std_mean[i]), alpha=0.3)

        axis_1.plot(std_true, std_error_to_gain_drop_error(std_true) * std_error[i] / gain_drop(nsb_rate) * 100,
                    label='$N_{bins} =$ %d' % (n_waveform * 50))

    # axis.set_xscale('log')
    axis.set_xlabel('STD [LSB]')
    axis.set_ylabel('gain drop []')
    axis.legend()

    axis_1.set_xlabel('STD [LSB]')
    axis_1.set_ylabel('gain drop error [$\%$]')
    axis_1.legend()

    fig = plt.figure()
    axis = fig.add_subplot(111)
    for j, nsb in enumerate(nsb_rate):

        if j % 4 == 0:

            axis.plot(n_waveforms * 50, std_error[..., j]/std_mean[..., j],
                      label='$f_{nsb} = $ %0.1f [MHz]' % (nsb * 1E3), linestyle='None', marker='o')

        # axis.plot(std_mean[i], std_to_gain_drop(std_mean[i]), alpha=0.3)

    # axis.set_xscale('log')
    axis.set_xlabel('$N_{bins}$')
    axis.set_ylabel('STD relative error []')
    axis.legend()

    plt.show()



