import logging
import sys

import h5py
import numpy as np
from tqdm import tqdm

from digicamtoy.container import Calibration_Container
from digicamtoy.core import Trace_Generator


def create_dataset(options):

    camera_parameters = Calibration_Container(filename=options.calibration_filename)
    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)
    log.debug('Reading calibration container %s \n' % options.calibration_filename)
    progress_bar = tqdm(total=len(options.file_list) * len(options.dc_level) * len(options.signal) * options.events_per_level)

    generator_parameters = {'start_time': options.photon_times[0],
                            'end_time': options.photon_times[1],
                            'sampling_time': options.photon_times[2],
                            'nsb_rate': None,
                            'mean_crosstalk_production': None,
                            'n_signal_photon': None,
                            'sig_poisson': options.poisson_signal,
                            'sigma_e': None,
                            'sigma_1': None,
                            'gain': None,
                            'baseline': None,
                            'gain_nsb_dependency': options.gain_nsb_dependency,
                            'time_signal': None,
                            'jitter_signal': None}

    simulation_parameters = {'start_time': options.photon_times[0],
                             'end_time': options.photon_times[1],
                             'sampling_time': options.photon_times[2],
                             'nsb_rate': np.zeros((camera_parameters.n_pixels, len(options.dc_level))),
                             'mean_crosstalk_production': camera_parameters.crosstalk['value'],
                             'n_signal_photon': options.signal,
                             'sig_poisson': options.poisson_signal,
                             'sigma_e': camera_parameters.electronic_noise['value'],
                             'sigma_1': camera_parameters.gain_smearing['value'],
                             'gain': camera_parameters.gain['value'],
                             'baseline': camera_parameters.baseline['value'],
                             'dark_rate': camera_parameters.dark_count_rate['value'],
                             'gain_nsb_dependency': options.gain_nsb_dependency,
                             'dc_dac': options.dc_level,
                             'time_signal': camera_parameters.time_signal['value'],
                             'jitter_signal': camera_parameters.time_jitter['value']}

    for file_index in options.file_list:

        log.debug('--|> Creating file %s \n' % (options.file_basename % file_index))

        hdf5 = h5py.File(options.output_directory + options.file_basename % file_index, 'x') # Create file, fail if exists
        camera_parameters_group = hdf5.create_group('simulation_parameters')

        for key, val in simulation_parameters.items():
            camera_parameters_group.create_dataset(key, data=val)
        for i, dc_level in enumerate(options.dc_level):

            for j, signal in enumerate(options.signal):

                generator_parameters['n_signal_photon'] = signal
                level_group = hdf5.create_group('dc_level_%d_ac_level_%d' % (i, j))

                generator = []

                for pixel_id in range(camera_parameters.n_pixels):

                    if not hasattr(options, 'nsb_rate'):
                        nsb = dc_led_fit_function(dc_level, a=camera_parameters.dc_led['value'][pixel_id][0], b=camera_parameters.dc_led['value'][pixel_id][1], c=camera_parameters.dc_led['value'][pixel_id][2]) * 1E3
                    else:
                        nsb = options.nsb_rate[i]

                    hdf5['simulation_parameters']['nsb_rate'][pixel_id, i] = nsb

                    generator_parameters['mean_crosstalk_production'] = camera_parameters.crosstalk['value'][pixel_id]
                    generator_parameters['sigma_e'] = camera_parameters.electronic_noise['value'][pixel_id]#/camera_parameters.gain['value'][pixel_id]
                    generator_parameters['sigma_1'] = camera_parameters.gain_smearing['value'][pixel_id]#/camera_parameters.gain['value'][pixel_id]
                    generator_parameters['gain'] = camera_parameters.gain['value'][pixel_id]
                    generator_parameters['baseline'] = camera_parameters.baseline['value'][pixel_id]
                    generator_parameters['nsb_rate'] = (nsb + camera_parameters.dark_count_rate['value'][pixel_id]) / 1E3
                    generator_parameters['time_signal'] = camera_parameters.time_signal['value'][pixel_id] * (generator_parameters['end_time'] - generator_parameters['start_time'])
                    generator_parameters['jitter_signal'] = camera_parameters.time_jitter['value'][pixel_id]

                    generator.append(Trace_Generator(**generator_parameters))



                log.debug('--|> Trace Generators created for dc_level (Dark count included) : %d and ac_level %d\n' % (i, j))
                event_number = 0
                data = []
                data_time = []
                while event_number < options.events_per_level:

                    try:
                        event = []
                        event_time = []
                        for pixel_id in range(camera_parameters.n_pixels):

                            generator[pixel_id].next()
                            event.append(generator[pixel_id].adc_count)
                            event_time.append(generator[pixel_id].cherenkov_time)

                        event_number += 1

                    except KeyboardInterrupt:
                        log.info('--|> MC production interrupted')
                        break
                    else:
                        data.append(event)
                        data_time.append(event_time)
                        progress_bar.update(1)

                data = np.array(data)
                data = np.rollaxis(data, 0, len(data.shape))
                data_time = np.array(data_time)

                level_group.create_dataset('data', data=data, dtype=np.uint16)
                level_group.create_dataset('time', data=data_time, dtype=np.float32)
                hdf5.flush()
        hdf5.close()
        log.info('--|> File %s.hdf5 saved to %s\n' % (options.file_basename % file_index, options.output_directory))

    return

def dc_led_fit_function(dc_level, a, b, c):

    if dc_level == 0:
        return 0.
    else:
        return a * np.exp(b * dc_level) + c