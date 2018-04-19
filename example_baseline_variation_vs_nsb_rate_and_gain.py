'''
Execute: python <path_to_this_script.py>

This creates a :
It takes a while (many minutes) and will create a plot, you'd have to
save manually to disk if you want to use it for a presentation or so.

'''
from digicamtoy.tracegenerator import NTraceGenerator
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

plt.figure()

N = 100
nsb_rates = np.logspace(-0.5, 1.2, 10)
results = []

for gain in tqdm(np.linspace(3, 5, 5)):
    for nsb_rate in nsb_rates:
        toy = NTraceGenerator(
            pulse_shape_file='/utils/pulse_SST-1M_pixel_0.dat',
            n_pixels=1,
            time_end=200,
            nsb_rate=nsb_rate,
            gain=gain,
            baseline=10,
            n_events=N,
        )
        baseline_std_deviation = np.array([
            event.adc_count[0].std()
            for event in toy
        ])

        results.append({
            'gain': gain,
            'nsb_rate': nsb_rate,
            'baseline_std_deviation_mean':
                baseline_std_deviation.mean(),
            'baseline_std_deviation_sde_of_mean':
                baseline_std_deviation.std() / np.sqrt(N),
        })

results = pd.DataFrame(results)

for gain, sub_results in results.groupby('gain'):
    plt.errorbar(
        x=sub_results.nsb_rate,
        y=sub_results.baseline_std_deviation_mean,
        yerr=sub_results.baseline_std_deviation_sde_of_mean,
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
