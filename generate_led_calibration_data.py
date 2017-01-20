from tracegenerator import TraceGenerator
import numpy as np
import matplotlib.pyplot as plt
from utils.progress_bar import print_progress_bar

def generate_mpe_spectrum(start_time=-50, end_time=50., sampling_time=4., nsb_rate=3. * 1E6 * 1E-9,
                 mean_crosstalk_production=0.08, n_signal_photon=3., sig_poisson= True, n_trace=500):

    i = 0

    amplitudes = np.zeros(n_trace)

    while i<n_trace:

        trace_object = TraceGenerator(start_time=start_time, end_time=end_time, sampling_time=sampling_time, nsb_rate=nsb_rate, mean_crosstalk_production=mean_crosstalk_production, n_signal_photon=n_signal_photon, sig_poisson=sig_poisson)
        adcs = trace_object.get_adc_count()



        trigger_time = trace_object.cherenkov_time
        bin_time = np.arange(start_time, end_time, sampling_time)
        trigger_bin = np.where((bin_time>=trigger_time - sampling_time/2.) * (bin_time<=trigger_time + sampling_time/2.))[0]
        trigger_window = int(3)

        amplitudes[i] = np.max(adcs[trigger_bin-trigger_window:trigger_bin+trigger_window+1])

        i += 1
        print_progress_bar(i, n_trace)

    bin_edges = np.arange(np.min(amplitudes), np.max(amplitudes)+2, 1)
    data = np.histogram(amplitudes, bins=bin_edges)[0]
    bin_centers = bin_edges[0:-1]

    mpe_spectrum = {'bin_centers': bin_centers, 'data': data}

    return mpe_spectrum

if __name__ == '__main__':

    n_trace = 10000

    mpe_spectrum = generate_mpe_spectrum(n_trace=n_trace)

    plt.figure()
    plt.errorbar(mpe_spectrum['bin_centers'], mpe_spectrum['data'], yerr=np.sqrt(mpe_spectrum['data']), marker='o', linestyle='None')
    plt.plot(mpe_spectrum['bin_centers'], mpe_spectrum['data'])
    plt.xlabel('ADC')
    plt.ylabel('count')
    plt.show()