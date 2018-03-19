import os
from optparse import OptionParser
import yaml
from yaml import CLoader
from digicamtoy.tracegenerator import NTraceGenerator
import h5py
import numpy as np
import copy
import logging
import humanize
from tqdm import tqdm
import datetime
import pandas as pd


if __name__ == '__main__':

    parser = OptionParser()

    # Job configuration (the only mandatory option)
    parser.add_option("-y", "--yaml_config", dest="yaml_config",
                      help="full path of the yaml configuration function",
                      default='options/ac_dc_scan.yaml')

    # Output level
    parser.add_option("-d", "--debug",
                      action="store_true", dest="debug", default=False,
                      help="move to debug")

    log = logging

    (options, args) = parser.parse_args()
    options_yaml = {}
    log.getLogger().setLevel(logging.INFO)

    if options.debug:
        log.getLogger().setLevel(logging.DEBUG)

    with open(options.yaml_config) as stream:

        options_yaml.update(yaml.load(stream, Loader=CLoader))

    for key, val in options_yaml.items():

        options.__dict__[key] = val

    for i in range(len(options.n_photon)*len(options.nsb_rate)):

        existing_file = False
        filename = options.output_directory + options.file_basename.format(i)
        if os.path.exists(filename):
            existing_file = True
            log.error('File {} already exists'.format(filename))
        if existing_file:
            exit()

    log.debug(vars(options))
    log.info('\t\t-|> Create the Monte Carlo data set')

    options_generator = copy.copy(options)
    file_number = 0
    n_pixels = options_generator.n_pixels
    n_bins = (options_generator.time_end - options_generator.time_start)
    n_bins = n_bins // options_generator.time_sampling
    events_per_level = options_generator.events_per_level

    data_shape = (events_per_level, n_pixels, n_bins)

    for nsb_rate in tqdm(options.nsb_rate,
                         leave=True,
                         total=len(options.nsb_rate),
                         desc='NSB'):

        options_generator.nsb_rate = nsb_rate

        for n_photon in tqdm(options.n_photon,
                             leave=False,
                             total=len(options.n_photon),
                             desc='p.e.'):

            options_generator.n_photon = n_photon

            if isinstance(n_photon, int) or isinstance(n_photon, float):
                n_photon = [[n_photon]] * n_pixels
                n_photon = np.atleast_2d(n_photon)

            else:
                n_photon = pd.DataFrame(n_photon).fillna(0).values

            filename = options.output_directory
            filename += options.file_basename.format(file_number)

            hdf5 = h5py.File(filename, 'w')
            config = hdf5.create_group('config')

            for key, val in vars(options_generator).items():

                config.create_dataset(key, data=val)

            data = hdf5.create_group('data')

            adc_count = np.zeros(data_shape, dtype=np.uint16)

            cherenkov_array_shape = (events_per_level,
                                     n_pixels,
                                     n_photon.shape[-1],
                                     )

            cherenkov_time = np.zeros(cherenkov_array_shape)
            cherenkov_photon = np.zeros(cherenkov_array_shape)

            for count, trace_generator in zip(tqdm(
                    range(options.events_per_level),
                    leave=False, desc='waveform'),
                    NTraceGenerator(**vars(options_generator))):

                adc_count[count] = trace_generator.adc_count
                cherenkov_time[count] = trace_generator.cherenkov_time
                cherenkov_photon[count] = trace_generator.cherenkov_photon

            log.info('\t\t-|> Saving data to {}'.format(filename))
            data.create_dataset('adc_count', data=adc_count,
                                dtype=np.uint16,
                                chunks=True,
                                compression="gzip")
            data.create_dataset('time',
                                data=cherenkov_time,
                                chunks=True,
                                compression="gzip")
            data.create_dataset('charge',
                                data=cherenkov_photon,
                                chunks=True,
                                compression="gzip")
            config.create_dataset('date',
                                  data=str(datetime.datetime.now()))
            hdf5.close()
            file_size = os.path.getsize(filename)
            file_size = humanize.naturalsize(file_size, binary=True)
            log.info('\t\t-|> File saved! Size = {}'.format(file_size))
            file_number += 1
