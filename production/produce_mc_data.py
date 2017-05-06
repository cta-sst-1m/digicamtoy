import h5py
import logging, sys
import numpy as np
from tqdm import tqdm
from core.trace_generator import Trace_Generator
from container.container import Calibration_Container


def create_dataset(options):


    camera_parameters = Calibration_Container(filename=options.calibration_filename)
    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)
    log.debug('Reading calibration container %s \n' %options.calibration_filename)
    pbar = tqdm(total=len(options.seed) * len(options.nsb_rate) * len(options.signal) * options.events_per_level)

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
                            'seed': None,
                            'gain_nsb_dependency': options.gain_nsb_dependency}

    simulation_parameters = {'start_time': options.photon_times[0],
                             'end_time': options.photon_times[1],
                             'sampling_time': options.photon_times[2],
                             'nsb_rate': options.nsb_rate,
                             'mean_crosstalk_production': camera_parameters.crosstalk['value'],
                             'n_signal_photon': options.signal,
                             'sig_poisson': options.poisson_signal,
                             'sigma_e': camera_parameters.electronic_noise['value'],
                             'sigma_1': camera_parameters.gain_smearing['value'],
                             'gain': camera_parameters.gain['value'],
                             'baseline': camera_parameters.baseline['value'],
                             'dark_rate' : camera_parameters.dark_count_rate['value'],
                             'seed': None,
                             'gain_nsb_dependency': options.gain_nsb_dependency}

    for seed in options.seed:

        generator_parameters['seed'] = seed
        simulation_parameters['seed'] = seed


        log.debug('--|> Creating file %s with seed %d\n' % (options.file_basename % seed, seed))

        hdf5 = h5py.File(options.output_directory + options.file_basename % seed, 'w')
        camera_parameters_group = hdf5.create_group('simulation_parameters')

        for key, val in simulation_parameters.items():
            camera_parameters_group.create_dataset(str(key), data=val)
        for i, nsb in enumerate(options.nsb_rate):

            for j, signal in enumerate(options.signal):

                generator_parameters['n_signal_photon'] = signal
                level_group = hdf5.create_group('dc_level_%d_ac_level_%d' % (i, j))

                generator = []

                for pixel_id in range(camera_parameters.n_pixels):

                    generator_parameters['mean_crosstalk_production'] = camera_parameters.crosstalk['value'][pixel_id]
                    generator_parameters['sigma_e'] = camera_parameters.electronic_noise['value'][pixel_id]/camera_parameters.gain['value'][pixel_id]
                    generator_parameters['sigma_1'] = camera_parameters.gain_smearing['value'][pixel_id]/camera_parameters.gain['value'][pixel_id]
                    generator_parameters['gain'] = camera_parameters.gain['value'][pixel_id]
                    generator_parameters['baseline'] = camera_parameters.baseline['value'][pixel_id]
                    generator_parameters['nsb_rate'] = (nsb + camera_parameters.dark_count_rate['value'][pixel_id]) / 1E3
                    generator_parameters['seed'] += 1

                    generator.append(Trace_Generator(**generator_parameters))

                log.debug('--|> Trace Generators created for dc_level (Dark count included) : %d and ac_level %d\n' % (i, j))
                event_number = 0
                data = []
                while event_number < options.events_per_level:

                    try:
                        event = []
                        for pixel_id in range(camera_parameters.n_pixels):

                            generator[pixel_id].next()
                            event.append(generator[pixel_id].adc_count)

                        event_number += 1

                    except KeyboardInterrupt:
                        log.info('--|> MC production interrupted')
                        break
                    else:
                        data.append(event)
                        pbar.update(1)

                data = np.array(data)
                data = np.rollaxis(data, 0, len(data.shape))
                level_group.create_dataset('data', data=data, dtype=np.uint16)
                hdf5.flush()
        hdf5.close()
        log.info('--|> File %s.hdf5 saved to %s\n' % (options.file_basename % seed, options.output_directory))

    return