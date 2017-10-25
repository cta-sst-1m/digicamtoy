from digicamtoy.core.Ntracegenerator import NTraceGenerator
from digicamtoy.core.tracegenerator import TraceGenerator
import numpy as np
import matplotlib.pyplot as plt


def test_sampling():

    param = {'time_start': 0, 'time_end': 100, 'time_sampling': 2}
    toy = NTraceGenerator(**param)
    toy.next()

    assert toy.adc_count.shape[-1] == (param['time_end'] - param['time_start']) // param['time_sampling']


def test_NTraceGenerator():

    n_toy_gen = NTraceGenerator(time_start = 0, time_end=200, time_sampling = 4, n_pixels = 1296,
                          nsb_rate = 0, crosstalk = 0.15, gain_nsb_dependency = True,
                          n_photon = 0, poisson = True, sigma_e = 0.8,
                          sigma_1 =  0.8, gain =  5.8,
                          baseline = 200., time_signal = 20, jitter = 0,
                          pulse_shape_file = '/utils/pulse_SST-1M_AfterPreampLowGain.dat', seed = None)

    toy_gen = TraceGenerator(start_time=0, end_time=196, sampling_time=4, nsb_rate=0.0,
                 mean_crosstalk_production=0.15, debug=False, gain_nsb_dependency=False, n_signal_photon=0.,
                 sig_poisson=True, sigma_e=0.8, sigma_1=0.8, gain=5.8, baseline=200., time_signal=20, jitter_signal=0,
                 pulse_shape_spline=None)

    n_events = 1000
    adc_1 = np.zeros((n_events, 50))
    adc_n = np.zeros((n_events, 1296, 50))
    for i in range(n_events):

        toy_gen.next()
        n_toy_gen.next()
        adc_1[i] = toy_gen.adc_count
        adc_n[i] = n_toy_gen.adc_count
        print(i)

    adc_1 = adc_1.ravel()
    adc_n = adc_n.ravel()

    plt.figure()
    plt.hist(adc_1, bins=np.arange(0, 300), normed=True, alpha=0.3, label='toy')
    plt.hist(adc_n, bins=np.arange(0, 300), normed=True, alpha=0.3, label='ntoy')
    plt.show()




if __name__ == '__main__':


    test_sampling()
    test_NTraceGenerator()


    #time_start = 0, time_end = 200, time_sampling = 4, n_pixels = 1296, nsb_rate = np.ones(1296) * 0.6,
    #crosstalk = np.ones(1296) * 0.08, gain_nsb_dependency = True, n_photon = np.zeros(1296), poisson = True,
    #sigma_e = np.ones(1296) * 0.8, sigma_1 = np.ones(1296) * 0.8, gain = np.ones(1296) * 5.8,
    #baseline = np.ones(1296) * 200., time_signal = np.ones(1296) * 20, jitter = np.zeros(1296),
    #pulse_shape_file = '/utils/pulse_SST-1M_AfterPreampLowGain.dat', seed = None