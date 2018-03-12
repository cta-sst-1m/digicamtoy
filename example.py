from digicamtoy.tracegenerator import NTraceGenerator
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    toy = NTraceGenerator(
        pulse_shape_file='/utils/pulse_SST-1M_pixel_0.dat',
        n_pixels=3,
        baseline=[0, 500, 0],
        n_photon=[[100, 133], [60, 60, 60], [100, 133]],
        time_signal=[[0, 100], [20, 50, 100], [0, 100]],
        gain=[10, 10, 29])

    t = np.arange(
        toy.time_start + toy.artificial_backward_time,
        toy.time_end,
        toy.time_sampling
    )

    for event in toy:

        print(event.count)

        plt.figure()
        plt.plot(t, event.adc_count[0], ls='steps-pre')
        plt.plot(t, event.adc_count[1], ls='steps-pre')
        plt.plot(t, event.adc_count[2], ls='steps-pre')
        plt.xlabel('t [ns]')
        plt.ylabel('[LSB]')
        plt.show()
