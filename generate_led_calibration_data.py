from trace_generator import Trace_Generator
import numpy as np
import matplotlib.pyplot as plt
from utils.progress_bar import print_progress_bar


def generate_traces(trace_object, n_trace):

    i = 0

    adcs = []


    while i<n_trace:

        trace_object.next()
        adcs.append(trace_object.get_adc_count())

        i = i + 1

    return adcs



def init_trace_generator(**kwargs):

    trace_object = Trace_Generator(**kwargs)

    return trace_object

if __name__ == '__main__':

    n_trace = 1000
    nsb_rate = 2.6 * 1E6 * 1E-9
    mu_xt = 0.06
    mu = 1. # np.arange(0, 50, 1)
    sigma_e = 0.1
    sigma_1 = 0.1
    sampling_time = 4
    gain = 5.6


    init_param = {'start_time' : -50, 'end_time': 50, 'sampling_time' : sampling_time, 'nsb_rate' : nsb_rate, 'mean_crosstalk_production' : mu_xt,
                  'n_signal_photon' : mu, 'sig_poisson': False, 'sigma_e' : sigma_e, 'sigma_1' : sigma_1, 'gain' : gain}

    generator = init_trace_generator(**init_param)

    traces = generate_traces(generator, n_trace=n_trace)
    #print(traces)
    amplitudes = np.max(traces, axis=1)

    print(amplitudes)
    print(min(amplitudes))
    print(max(amplitudes))


    plt.figure()
    plt.hist(amplitudes, bins=np.arange(np.min(amplitudes), np.max(amplitudes) + 2, 1), align='left')
    plt.xlabel('ADC')
    plt.ylabel('count')
    plt.show()