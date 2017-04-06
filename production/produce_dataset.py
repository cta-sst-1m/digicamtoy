import h5py
import logging,sys
import numpy as np
from tqdm import tqdm
from core.trace_generator import Trace_Generator
import matplotlib.pyplot as plt


def create_dataset(options):

    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)
    pbar = tqdm(total=len(options.seed) * len(options.nsb_rate) * len(options.signal) * options.events_per_level)

    generator_parameters = {'start_time': options.photon_times[0], 'end_time': options.photon_times[1],
                 'sampling_time': options.photon_times[2], 'nsb_rate': options.nsb_rate[0],
                 'mean_crosstalk_production': options.crosstalk,
                 'n_signal_photon': options.signal[0], 'sig_poisson': options.poisson_signal,
                 'sigma_e': options.sigma_e/options.gain,
                 'sigma_1': options.sigma_1/options.gain, 'gain': options.gain, 'baseline': options.baseline,
                 'seed': options.seed[0],
                 'gain_nsb_dependency': options.gain_nsb_dependency}

    for seed in options.seed:

        generator_parameters['seed'] = seed

        log.info('--|> Creating file %s with seed %d ' % (options.file_basename % seed, seed))

        f = h5py.File(options.output_directory + options.file_basename % seed, 'w')


        for i, nsb in enumerate(options.nsb_rate):

            generator_parameters['nsb_rate'] = nsb / 1E3

            for j, signal in enumerate(options.signal):

                generator_parameters['n_signal_photon'] = signal
                level_group = f.create_group('dc_level_%d_ac_level_%d' % (i, j))
                generator = Trace_Generator(**generator_parameters)
                generator.next()

                for key, val in generator.__dict__.items():
                    level_group.create_dataset(str(key), data=val)

                traces = []
                event_number = 0
                while event_number < options.events_per_level:
                    generator.next()
                    traces.append(generator.adc_count)
                    event_number += 1
                    pbar.update(1)

                traces = np.array(traces)
                level_group.create_dataset('trace', data=traces)  # , chunks=False)

        log.info('--|> File %s.hdf5 saved to %s' % (options.file_basename % seed, options.output_directory))

        f.close()
    return

def display(options):

    file = h5py.File(options.output_directory + options.file_basename % options.seed[0], 'r')

    trace = file['dc_level_%d_ac_level_%d' %(0, 0)]['trace']
    trace = np.array(trace).ravel()

    fig = plt.figure()
    axis = fig.add_subplot(111)
    hist = axis.hist(trace, bins=np.arange(np.min(trace), np.max(trace), 1))
    axis.set_yscale('log')
    axis.set_ylabel('count')
    axis.set_xlabel('ADC')

    np.savez(options.output_directory + 'hist_dark', bin=hist[1][0:-1], count=hist[0])


    return