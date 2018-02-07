import os
from optparse import OptionParser
import yaml
from digicamtoy.core.Ntracegenerator import NTraceGenerator
import h5py
import numpy as np
import copy
import logging
import sys
import tqdm
import datetime
import pandas as pd


if __name__ == '__main__':

    parser = OptionParser()

    # Job configuration (the only mandatory option)
    parser.add_option("-y", "--yaml_config", dest="yaml_config",
                      help="full path of the yaml configuration function",
                      default='options/ac_dc_scan.yaml')

    # Output level
    parser.add_option("-v", "--verbose",
                      action="store_false", dest="verbose", default=True,
                      help="move to debug")

    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)

    (options, args) = parser.parse_args()
    options_yaml = {}
    log.setLevel(0 if options.verbose else 20)

    with open(options.yaml_config) as stream:

        options_yaml.update(yaml.load(stream))

    for key, val in options_yaml.items():

        options.__dict__[key] = val

    for i in range(len(options.n_photon)*len(options.nsb_rate)):

        existing_file = False
        if os.path.exists(options.output_directory + options.file_basename.format(i)):
            existing_file = True
            log.error('File {} already exists'.format(options.output_directory + options.file_basename.format(i)))
        if existing_file:
            exit()

    progress_bar = tqdm.tqdm(total=len(options.nsb_rate) * len(options.n_photon) * options.events_per_level)
    log.debug(vars(options))
    log.info('\t\t-|> Create the Monte Carlo data set')

    options_generator = copy.copy(options)
    file_number = 0
    n_pixels = options_generator.n_pixels
    n_bins = (options_generator.time_end - options_generator.time_start) // options_generator.time_sampling
    events_per_level = options_generator.events_per_level
    data_shape = (n_pixels, n_bins, events_per_level)

    for nsb_rate in options.nsb_rate:

        options_generator.nsb_rate = nsb_rate

        for n_photon in options.n_photon:

            options_generator.n_photon = n_photon

            if isinstance(n_photon, int) or isinstance(n_photon, float):
                n_photon = [[n_photon]] * n_pixels
                n_photon = np.atleast_2d(n_photon)

            else:
                n_photon = pd.DataFrame(n_photon).fillna(0).values

            hdf5 = h5py.File(options.output_directory + options.file_basename.format(file_number), 'w')
            config = hdf5.create_group('config')

            for key, val in vars(options_generator).items():

                config.create_dataset(key, data=val)

            data = hdf5.create_group('data')

            # adc_count = data.create_dataset('adc_count', (n_pixels, n_bins, events_per_level), dtype=np.uint16, compression="gzip")
            # cherenkov_time = data.create_dataset('time', n_photon.shape + (events_per_level, ), compression="gzip")
            # cherenkov_photon = data.create_dataset('charge', n_photon.shape + (events_per_level, ), compression="gzip")

            adc_count = np.zeros(data_shape, dtype=np.uint16)
            cherenkov_time = np.zeros((n_pixels, n_photon.shape[-1], events_per_level))
            cherenkov_photon = np.zeros((n_pixels, n_photon.shape[-1], events_per_level))

            for count, trace_generator in zip(range(options.events_per_level), NTraceGenerator(**vars(options_generator))):

                adc_count[..., count] = trace_generator.adc_count
                cherenkov_time[..., count] = trace_generator.cherenkov_time
                cherenkov_photon[..., count] = trace_generator.cherenkov_photon

                progress_bar.update(1)

            log.info('\t\t-|> Saving data to {}'.format(options.file_basename.format(file_number)))
            data.create_dataset('adc_count', data=adc_count, dtype=np.uint16, compression="gzip")
            data.create_dataset('time', data=cherenkov_time, compression="gzip")
            data.create_dataset('charge', data=cherenkov_photon, compression="gzip")
            config.create_dataset('date', data=str(datetime.datetime.now()))
            hdf5.close()
            progress_bar.update(1)
            log.info('\t\t-|> Done !!!')
            file_number += 1
