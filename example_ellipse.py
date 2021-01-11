from digicamtoy.generator.image import EllipseGenerator
import matplotlib.pyplot as plt
import numpy as np
from digicampipe.instrument.camera import DigiCam
from digicampipe.visualization.plot import plot_array_camera
import matplotlib.pyplot as plt

if __name__ == '__main__':

    n_events = 1
    n_images = 100

    for _ in range(n_images):

        sigma = np.random.uniform(23.4, 100, size=2)
        true_image_parameters = {'x_cm': np.random.uniform(-400, 400),
                                 'y_cm': np.random.uniform(-400, 400),
                                 'width': sigma.min(),
                                 'length': sigma.max(),
                                 'psi': np.random.uniform(0, 2*np.pi),
                                 'size': np.random.uniform(0, 1000),
                                 'time_cm': np.random.uniform(10, 50),
                                 'velocity': np.random.choice([-1, 1]) * np.random.uniform(0.008, 0.02)}
        true_image_parameters['size'] = 5000
        true_image_parameters['time_cm'] = 40
        digicam_parameters = {'time_start': 0,
                              'time_end': 200,
                              'time_sampling': 4,
                              'n_pixels': 1296,
                              'nsb_rate': 0.00,
                              'crosstalk': 0.08,
                              'gain_nsb': True,
                              'poisson': True,
                              'sigma_e': 0.8,
                              'sigma_1': 0.08,
                              'gain': 1.0,
                              'baseline': 200,
                              'time_signal': 20,
                              'jitter': 0,
                              'pulse_shape_file': 'utils/pulse_SST-1M_pixel_0.dat',
                              'sub_binning': 0,
                              'n_events': n_events,
                              'voltage_drop': True,}

        toy = EllipseGenerator(**true_image_parameters,
                               geometry=DigiCam.geometry,
                               **digicam_parameters,
                               )
        t = np.arange(digicam_parameters['time_start'],
                      digicam_parameters['time_end'],
                      digicam_parameters['time_sampling'])
        for event in toy:

            print(event.count)

            waveform = event.adc_count - event.true_baseline[:, None]

            """
            true_pe = event.n_photon[:, 0]

            pixel_with_pe = np.argsort(true_pe)[-10:]
            plt.figure()

            for pixel in pixel_with_pe:
                plt.step(t, waveform[pixel])

            plt.xlabel('time [ns]')
            plt.ylabel('[LSB]')

            plot_array_camera(waveform.mean(axis=-1),
                              label='Integrated waveform [LSB]')
            plot_array_camera(true_pe,
                              label='Photo-electrons [p.e.]')
            plot_array_camera(event.time_signal[:, 0],
                              label='Time of arrival [ns]')

            plt.show()
            """