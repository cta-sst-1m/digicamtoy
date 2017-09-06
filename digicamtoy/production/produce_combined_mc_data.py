import h5py
import logging
import sys
from shutil import copyfile


def create_dataset(options):

    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)
    log.info('Merging %s and %s \n' % (options.file_1, options.file_2))
    copyfile(options.output_directory + options.file_1, options.output_directory + options.file_basename)

    hdf5_1 = h5py.File(options.output_directory + options.file_1, 'r')
    hdf5_2 = h5py.File(options.output_directory + options.file_2, 'r')
    hdf5 = h5py.File(options.output_directory + options.file_basename, 'r+')

    n_ac_level = len(hdf5_1['simulation_parameters']['n_signal_photon'])
    n_dc_level = hdf5_2['simulation_parameters']['nsb_rate'].shape[-1]
    data_shape_1 = hdf5_1['dc_level_0_ac_level_0']['data'].shape
    data_shape_2 = hdf5_2['dc_level_0_ac_level_0']['data'].shape

    if n_ac_level != len(hdf5_2['simulation_parameters']['n_signal_photon']) or n_dc_level != hdf5_2['simulation_parameters']['nsb_rate'].shape[-1]:
        log.error('Could not merge : files contain not the same amount of groups\n')
        exit()

    elif len(options.pixel_list) != data_shape_2[0]:

        log.error('Could not merge : not enough pixels in %s (%d, %d)\n' % (options.file_2, len(options.pixel_list), hdf5_2['dc_level_0_ac_level_0']['data'].shape[0]))
        exit()

    elif data_shape_1[-1] != data_shape_2[-1]:

        log.error('Could not merge : traces to merge have different shapes (%d, %d)\n' % (data_shape_1[-1], data_shape_2[-1]))
        exit()

    else:

        for ac_level in range(n_ac_level):
            for dc_level in range(n_dc_level):
                group_name = 'dc_level_%d_ac_level_%d' % (dc_level, ac_level)

                print( hdf5[group_name]['data'][options.pixel_list, :, :])
                print( hdf5[group_name]['data'][options.pixel_list, :, :])
                print(hdf5_2[group_name]['data'])

                hdf5[group_name]['data'][options.pixel_list, :, :] = hdf5_2[group_name]['data']

        hdf5.close()
        hdf5_1.close()
        hdf5_2.close()

        log.info('File merged to %s' % options.file_basename)