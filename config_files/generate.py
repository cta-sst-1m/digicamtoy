import yaml
import numpy as np
import os


def add_standard_sipm_parameters(parameters):

    n_pixels = parameters['n_pixels']

    parameters['sigma_e'] = [0.8] * n_pixels
    parameters['sigma_1'] = [0.8] * n_pixels
    parameters['gain'] = [5.8] * n_pixels
    parameters['baseline'] = [200] * n_pixels
    parameters['time_signal'] = [20] * n_pixels
    parameters['jitter'] = [1.155] * n_pixels
    parameters['crosstalk'] = [0.08] * n_pixels


def add_standard_camera_parameters(parameters):

    parameters['time_start'] = 0
    parameters['time_end'] = 368
    parameters['time_sampling'] = 4
    parameters['n_pixels'] = 1296
    parameters['gain_nsb'] = True
    parameters['poisson'] = True
    parameters['pulse_shape_file'] = 'utils/pulse_SST-1M_pixel_0.dat'


def add_light_parameters(parameters, nsb_rate, n_photon):

    n_pixels = parameters['n_pixels']

    parameters['nsb_rate'] = [float(nsb_rate)] * n_pixels
    parameters['n_photon'] = [float(n_photon)] * n_pixels


if __name__ == '__main__':

    params = dict()
    params['output_directory'] = '/sst1m/MC/digicamtoy/'
    params['events_per_level'] = 20000

    add_standard_camera_parameters(params)
    add_standard_sipm_parameters(params)

    output_file_name = 'nsb_{}.hdf5'
    config_file_path = 'commissioning/nsb_{}.yml'

    nsb_rates = np.linspace(0, 1, num=30)

    for i, nsb_rate in enumerate(nsb_rates):

        add_light_parameters(params, nsb_rate, n_photon=0)
        params['file_basename'] = output_file_name.format(i)

        with open(config_file_path.format(i), mode='w') as file:

            yaml.dump(params, file)
