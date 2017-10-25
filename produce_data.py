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
    log.info(vars(options))
    log.info('\t\t-|> Create the Monte Carlo data set')

    options_generator = copy.copy(options)
    file_number = 0

    for nsb_rate in options.nsb_rate:

        options_generator.nsb_rate = nsb_rate

        for n_photon in options.n_photon:

            options_generator.n_photon = n_photon

            hdf5 = h5py.File(options.output_directory + options.file_basename.format(file_number), 'w')
            config = hdf5.create_group('config')

            for key, val in vars(options_generator).items():

                config.create_dataset(key, data=val)

            data = hdf5.create_group('data')
            adc_count = []
            cherenkov_time = []
            cherenkov_photon = []

            for count, trace_generator in zip(range(options.events_per_level), NTraceGenerator(**vars(options_generator))):

                adc_count.append(trace_generator.adc_count)
                cherenkov_time.append(trace_generator.cherenkov_time)
                cherenkov_photon.append(trace_generator.cherenkov_photon)

                progress_bar.update(1)

            adc_count = np.array(adc_count)
            adc_count = np.rollaxis(adc_count, 0, len(adc_count.shape))
            data.create_dataset('adc_count', data=adc_count, dtype=np.uint16)

            cherenkov_time = np.array(cherenkov_time)
            cherenkov_time = np.rollaxis(cherenkov_time, 0, len(cherenkov_time.shape))
            data.create_dataset('time', data=cherenkov_time)

            cherenkov_photon = np.array(cherenkov_photon)
            cherenkov_photon = np.rollaxis(cherenkov_photon, 0, len(cherenkov_photon.shape))
            data.create_dataset('charge', data=cherenkov_photon)

            hdf5.close()
            file_number += 1
