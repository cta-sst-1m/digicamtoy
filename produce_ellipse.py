import os
from optparse import OptionParser
import h5py
import logging
import humanize
from tqdm import tqdm
import datetime
from digicamtoy.generator.image import EllipseGenerator
import numpy as np
from digicampipe.instrument.camera import DigiCam


if __name__ == '__main__':

    parser = OptionParser()

    # Job configuration (the only mandatory option)
    parser.add_option("--n_events", dest="n_events",
                      help="Number of identical events",
                      default=1)
    parser.add_option("--n_images", dest="n_images",
                      help="Number of events",
                      default=1000)
    parser.add_option("--output", dest='output', help='output file',
                      default='test.hdf5')
    parser.add_option("--nsb", dest='nsb', help='NSB rate in GHz',
                      default=.0)

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

    n_events = int(options.n_events)
    n_images = int(options.n_images)
    filename = str(options.output)
    nsb_rate = float(options.nsb)

    if os.path.exists(filename):
        log.error('File {} already exists'.format(filename))
        exit()

    digicam_parameters = {'time_start': 0,
                          'time_end': 200,
                          'time_sampling': 4,
                          'n_pixels': 1296,
                          'nsb_rate': nsb_rate,
                          'crosstalk': 0.08,
                          'gain_nsb': True,
                          'poisson': True,
                          'sigma_e': 1,
                          'sigma_1': 1,
                          'gain': 10,
                          'baseline': 200,
                          'time_signal': 20,
                          'jitter': 0,
                          'pulse_shape_file': 'utils/pulse_SST-1M_pixel_0.dat',
                          'sub_binning': 0,
                          'n_events': n_events,
                          'voltage_drop': True,}

    n_pixels = digicam_parameters['n_pixels']
    n_bins = (digicam_parameters['time_end'] - digicam_parameters['time_start'])
    n_bins = n_bins // digicam_parameters['time_sampling']
    data_shape = (n_images * n_events, n_pixels, n_bins)
    hdf5 = h5py.File(filename, 'w')
    config = hdf5.create_group('config')
    for key, val in digicam_parameters.items():
        config.create_dataset(key, data=val)
    data_group = hdf5.create_group('data')
    mc_group = hdf5.create_group('mc')
    adc_count = np.zeros(data_shape, dtype=np.uint16)
    cherenkov_time = np.zeros(data_shape[:-1])
    cherenkov_photon = np.zeros(data_shape[:-1])
    baseline = np.zeros(data_shape[:-1])
    mc_params = {'x_cm': [], 'y_cm': [], 'width': [], 'length': [], 'psi': [],
                 'size': [], 'time_cm': [], 'velocity': []}


    event_id = 0

    log.info('\t\t-|> Create the Monte Carlo data set')

    for _ in tqdm(range(n_images), desc='# image'):

        sigma_l = np.random.uniform(23.4, 100)
        sigma_w = np.random.uniform(0.3, 0.7) * sigma_l
        density = np.random.uniform(20, 100)
        size = density * sigma_l * sigma_w / DigiCam.geometry.pix_area[0]

        true_image_parameters = {'x_cm': np.random.uniform(-400, 400),
                                 'y_cm': np.random.uniform(-400, 400),
                                 'width': sigma_w,
                                 'length': sigma_l,
                                 'psi': np.random.uniform(0, np.pi),
                                 'size': size,
                                 'time_cm': np.random.uniform(10, 30),
                                 'velocity': np.random.choice(
                                     [-1, 1]) * np.random.uniform(0.01,
                                                                  0.1)}
        toy = EllipseGenerator(**true_image_parameters,
                               geometry=DigiCam.geometry,
                               verbose=False,
                               **digicam_parameters)

        for event, _ in zip(toy, range(n_events)):

            adc_count[event_id] = event.adc_count
            cherenkov_time[event_id] = event.cherenkov_time[:, 0]
            cherenkov_photon[event_id] = event.cherenkov_photon[:, 0]
            baseline[event_id] = event.true_baseline

            for key, val in true_image_parameters.items():

                mc_params[key].append(val)

            event_id += 1

    log.info('\t\t-|> Saving data to {}'.format(filename))

    for key, val in mc_params.items():
        mc_group.create_dataset(key, data=np.array(val))

    data_group.create_dataset('adc_count', data=adc_count,
                              dtype=np.uint16,
                              chunks=True,
                              compression="gzip")
    data_group.create_dataset('time',
                              data=cherenkov_time,
                              chunks=True,
                              compression="gzip")
    data_group.create_dataset('charge',
                              data=cherenkov_photon,
                              chunks=True,
                              compression="gzip")
    data_group.create_dataset('true_baseline',
                              data=baseline,
                              compression="gzip")
    config.create_dataset('date',
                          data=str(datetime.datetime.now()))
    hdf5.close()
    file_size = os.path.getsize(filename)
    file_size = humanize.naturalsize(file_size, binary=True)
    log.info('\t\t-|> File saved! Size = {}'.format(file_size))
