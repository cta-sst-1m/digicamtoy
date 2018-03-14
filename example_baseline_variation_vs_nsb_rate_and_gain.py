from digicamtoy.tracegenerator import NTraceGenerator
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.figure()

N = 3000
nsb_rates = np.logspace(-0.5, 1.2, 30)
for gain in np.linspace(3, 5, 5):
    means = []
    stds = []
    gains = []

    for nsb_rate in nsb_rates:
        print(nsb_rate)
        toy = NTraceGenerator(
            pulse_shape_file='/utils/pulse_SST-1M_pixel_0.dat',
            n_pixels=1,
            time_end=200,
            nsb_rate=nsb_rate,
            gain=gain,
            baseline=10,
        )
        gains.append(toy.gain[0])
        M = []
        S = []
        for i, event in enumerate(toy):
            if i > N:
                break
            M.append(event.adc_count[0].mean())
            S.append(event.adc_count[0].std())
        means.append(M)
        stds.append(S)

    means = np.array(means)
    stds = np.array(stds)


    """
    plt.errorbar(
        x=nsb_rates,
        y=means.mean(axis=1) / means.mean(axis=1).max(),
        yerr=means.std(axis=1) / (np.sqrt(N) * means.mean(axis=1).max()),
        fmt='.:',
        label='means',
    )
    plt.semilogx(nsb_rates, gains / np.max(gains), '.:', label='gains')
    """

    plt.errorbar(
        x=nsb_rates,
        y=stds.mean(axis=1),
        #yerr=stds.std(axis=1) / (np.sqrt(N) * stds.mean(axis=1).max()),
        fmt='.:',
        label='gain: {}'.format(gain),
    )

ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
plt.xlabel('nsb rate [GHz]')
plt.ylabel('std-dev of baseline')

plt.grid()
plt.legend()

plt.show()
