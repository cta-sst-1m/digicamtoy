from trace_generator import Trace_Generator
import numpy as np
import matplotlib.pyplot as plt
from utils.progress_bar import print_progress_bar


def generate_mpe_spectrum(start_time=-50, end_time=50., sampling_time=4., nsb_rate=3. * 1E6 * 1E-9,
                 mean_crosstalk_production=0.08, n_signal_photon=3., sig_poisson= True, n_trace=500, sigma_e=0.080927083, sigma_1=0.092927083, gain=5.6):

    i = 0

    adcs = []

    while i<n_trace:

        trace_object = Trace_Generator(start_time=start_time, end_time=end_time, sampling_time=sampling_time, nsb_rate=nsb_rate, mean_crosstalk_production=mean_crosstalk_production, n_signal_photon=n_signal_photon, sig_poisson=sig_poisson, sigma_e=sigma_e, sigma_1=sigma_1, gain=gain)
        adcs.append(trace_object.get_adc_count())

        i = i + 1

    return adcs



if __name__ == '__main__':

    n_trace = 100
    nsb_rate = 2.6 * 1E6 * 1E-9
    mu_xt = 0.06
    mu = 1.
    sigma_e = 0.1
    sigma_1 = 0.1
    gain = 5.6

    mpe_spectrum = generate_mpe_spectrum(n_trace=n_trace, nsb_rate=nsb_rate, sigma_e=sigma_e, sigma_1=sigma_1, mean_crosstalk_production=mu_xt, n_signal_photon=mu, gain=gain)

    plt.figure()
    plt.errorbar(mpe_spectrum['bin_centers'], mpe_spectrum['data'], yerr=np.sqrt(mpe_spectrum['data']), marker='o', linestyle='None')
    plt.plot(mpe_spectrum['bin_centers'], mpe_spectrum['data'])
    plt.xlabel('ADC')
    plt.ylabel('count')
    plt.show()