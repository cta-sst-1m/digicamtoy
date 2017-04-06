import numpy as np
import h5py
from optparse import OptionParser
from trace_generator import Trace_Generator
import logging, sys
from tqdm import tqdm

def createChunk(options,tr):
    """
    Create a data chunck
    :param tr:
    :param chunkSize:
    :return:
    """
    chunkSize = options.batch_max
    ii = 0
    labels,traces = None,None


    while ii < chunkSize:

        if options.nsb_range[2]!=0:

            tr.nsb_rate = ((np.random.random_sample() * (options.nsb_range[1]-options.nsb_range[0]))+options.nsb_range[0]) * 1e-3

        else:

            tr.nsb_rate = options.nsb_range[0]

        if options.photon_range[2]!=0:

            tr.n_signal_photon = int((np.random.random_sample() * (options.photon_range[1]-options.photon_range[0]))+options.photon_range[0])

        else:

            tr.n_signal_photon = options.photon_range[0]


        tr.next()
        ravelled_arrival = []
        for k,t0 in enumerate(tr.photon_arrival_time):
            for kk in range(int(round(tr.photon[k]))):
                ravelled_arrival.append(t0)

        n_photons,axis  = np.histogram(ravelled_arrival,
                                  bins=np.arange(options.photon_times[0],
                                                 options.photon_times[1]+options.photon_times[2]+options.target_segmentation,
                                                 options.target_segmentation))

        n_adcs = tr.adc_count.repeat(int(np.round(options.photon_times[2]/options.target_segmentation)))
        if type(traces).__name__ == 'ndarray':
            traces = np.append(traces,n_adcs.reshape((1,)+n_adcs.shape),axis=0)
            labels = np.append(labels,n_photons.reshape((1,)+n_photons.shape),axis=0)
        else:
            traces = n_adcs.reshape((1,)+n_adcs.shape)
            labels = n_photons.reshape((1,)+n_photons.shape)
        ii+=1
    return traces.reshape(traces.shape+(1,)),labels.reshape(labels.shape+(1,))



def createFile( options,tr , filename):
    """
    Create a large HDF5 data file
    """
    chunckSize, finalSize = options.batch_max,options.evt_max

    print("Progress {:2.1%}".format(0.), end="\r")

    chunks = createChunk(options,tr)

    f = h5py.File(filename+'.hdf5', 'w')

    level_group = f.create_group('level')
    data_group = f.create_group('data')
    param_group = f.create_group('param')

    level_group.create_dataset('traces', data=chunks[0], chunks=True, maxshape=(None,chunks[0].shape[1],chunks[0].shape[2]))
    data_group.create_dataset('traces', data=chunks[0], chunks=True, maxshape=(None,chunks[0].shape[1],chunks[0].shape[2]))
    data_group.create_dataset('labels', data=chunks[1], chunks=True,
                              maxshape=(None, chunks[1].shape[1], chunks[1].shape[2]))
    for key, val in tr.__dict__.items():

        param_group.create_dataset(key, data=val)


    traces_dataset = f['data']['traces']
    labels_dataset = f['data']['labels']

    #print(traces_dataset)

    nChunks = finalSize // chunckSize
    for i in range(nChunks):
        print("Progress {:2.1%}".format(float(i*options.batch_max) / options.evt_max), end="\r")
        chunks = createChunk(options,tr)
        newshape = [traces_dataset.shape[0] + chunks[0].shape[0],chunks[0].shape[1],chunks[0].shape[2]]
        traces_dataset.resize(newshape)
        newshape_label = [labels_dataset.shape[0] + chunks[1].shape[0],chunks[1].shape[1],chunks[1].shape[2]]
        labels_dataset.resize(newshape_label)
        traces_dataset[-chunks[0].shape[0]:] = chunks[0]
        labels_dataset[-chunks[1].shape[0]:] = chunks[1]

    f.close()

def create_mc_data_set(options):

    seeds = options.seeds
    levels = options.levels

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=len(seeds)*len(levels)*options.events_per_level)
    #tqdm_out = TqdmToLogger(log, level=logging.INFO)

    for seed in seeds:

        print('--|> Opening file %s ' % (options.filename % seed + '.hdf5'))

        f = h5py.File(options.directory + options.filename % seed + '.hdf5', 'w')

        for level in levels:

            if options.sig_poisson:
                mu = np.polyval(options.led_calib_const, level*5)
            else:
                mu = 0
                #mu = level




            print('--|> Creating goup for level %d : XT = %0.2f, dark_rate = %0.2f [MHz], mu = %0.2f [p.e.]' %(level, options.mean_crosstalk, options.nsb[level] * 1E3, mu))

            level_group = f.create_group('level_%d' % level)


            # Create the trace generator
            toy_param = {'start_time' :options.photon_times[0], 'end_time': options.photon_times[1], 'sampling_time' : options.photon_times[2], 'nsb_rate' : options.nsb[level], 'mean_crosstalk_production' : options.mean_crosstalk,
                      'n_signal_photon' : mu, 'sig_poisson': options.sig_poisson, 'sigma_e' : options.sigma_e, 'sigma_1': options.sigma_1, 'gain' : options.gain, 'baseline': options.baseline, 'seed': seed, 'gain_nsb_dependency' : options.gain_nsb_dependency}
            toy = Trace_Generator(**toy_param)
            toy.next()

            for key, val in toy.__dict__.items():
                level_group.create_dataset(str(key), data=val)

            traces = []
            i = 0
            while i<options.events_per_level:
                toy.next()
                traces.append(toy.adc_count.tolist())
                i += 1
                pbar.update(1)

            traces = np.array(traces)
            trace_set = level_group.create_dataset('trace', data=traces)#, chunks=False)


        f.close()

        print('--|> File %s.hdf5 saved to %s' %(options.filename % seed, options.directory))


if __name__ == '__main__':

    # Job configuration
    parser = OptionParser()


    parser.add_option("-d", "--directory", dest="directory",
                      help="Output directory", default="./data_calibration_cts/")

    parser.add_option("-f", "--filename", dest="filename",
                      help="Output file name", default="xt_0_dark_0_seed_%d")

    parser.add_option("--seeds", dest="seeds",
                      help="seeds", default=[0], type=float)

    parser.add_option("--levels", dest="levels",
                      help="levels (# p.e. to simulate)", default=[0], type=int)#, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], type=int)

    parser.add_option("--LED_calib_const", dest='led_calib_const',
                      help='LED calibration constants for 4th degree polynomial (ax^4+bx^3+cx^2+dx+e)', default=[11* 1E-8, 0., 0., 0., 0.], type=float)

    parser.add_option("--events_per_level", dest="events_per_level",
                      help="number of traces per level", default=1E4, type=int)

    parser.add_option("-b", "--nsb", dest="nsb",
                      help="NSB in GHz", default=[0.003], type=float)

    parser.add_option("-x", "--crosstalk", dest="mean_crosstalk",
                      help="Mean number of crosstalk produced per discharged microcell", default=0.06, type=float)

    parser.add_option("-g", "--gain", dest="gain",
                      help="FADC gain in ADC/p.e.", default=5.6, type=float)

    parser.add_option("--gain_nsb", dest="gain_nsb_dependency",
                      help="gain depedency with nsb", action='store_true', default=False)

    parser.add_option("--sigma_e", dest="sigma_e",
                      help="electronic noise in p.e.", default=0.86/5.6, type=float)

    parser.add_option("--sigma_1", dest="sigma_1",
                      help="SiPM gain smearing in p.e.", default=0.86/5.6)

    parser.add_option("--photon_times", dest="photon_times",
                      help="arrival time range", default=[0, 100000, 4], nargs=3, type=float)

    parser.add_option("--baseline", dest="baseline",
                     help="user set baseline", default=500., type=float)

    parser.add_option("--poisson_signal", dest="sig_poisson",
                      help="if 1 the signal is generated according to a Poisson distribution", default=False, type=int)

    (options, args) = parser.parse_args()


    #if ',' in options.photon_range: options.photon_range = [float(n) for n in options.photon_range.split(',')]
    #if ',' in options.nsb_range: options.nsb_range = [float(n) for n in options.nsb_range.split(',')]
    #options.photon_times = [float(n) for n in options.photon_times.split(',')]
    #options.levels = [float(n) for n in options.levels.split(',')]
    #options.seeds = [float(n) for n in options.seeds.split(',')]

    #print(options)
    create_mc_data_set(options)

    # # Job configuration
    # parser = OptionParser()
    # parser.add_option("-n", "--evt_max", dest="evt_max",
    #                   help="maximal number of events", default=10000, type=int)
    #
    # parser.add_option("--batch_max", dest="batch_max",
    #                   help="maximal number of events for batch in memory", default=100, type=int)
    #
    # parser.add_option("-d", "--directory", dest="directory",
    #                   help="Output directory", default="./")
    #
    # parser.add_option("-f", "--filename", dest="filename",
    #                   help="Output file name", default="test")
    #
    # parser.add_option("-p", "--photon_range", dest="photon_range",
    #                   help="range of signal photons", nargs=3, type=float, default=[0., 40., 1.])
    #
    # parser.add_option("-b", "--nsb_range", dest="nsb_range",
    #                   help="range of NSB", default=[0.,200., 20.], nargs=3, type=float)
    #
    # parser.add_option("-x", "--crosstalk", dest="mean_crosstalk",
    #                   help="Mean number of crosstalk produced per discharged microcell", default=0.06, type=float)
    #
    # parser.add_option("-g", "--gain", dest="gain",
    #                   help="FADC gain in ADC/p.e.", default=5.6)
    #
    # parser.add_option("--sigma_e", dest="sigma_e",
    #                   help="electronic noise in p.e.", default=0.86/5.6)
    #
    # parser.add_option("--sigma_1", dest="sigma_1",
    #                   help="SiPM gain smearing in p.e.", default=0.48/5.6)
    #
    # parser.add_option("--photon_times", dest="photon_times",
    #                   help="arrival time range", default=[-50, 50, 4], nargs=3, type=float)
    #
    # parser.add_option("--baseline", dest="baseline",
    #                  help="user set baseline", default=2010., type=float)
    #
    # parser.add_option("--target_segmentation", dest="target_segmentation",
    #                   help="arrival time range", default=4, type=float)
    #
    # parser.add_option("--poisson_signal", dest="sig_poisson",
    #                   help="if 1 the signal is generated according to a Poisson distribution", default=False, type=int)
    #
    # #print(parser.rargs)
    # #print(parser.nargs)
    # #print(parser)
    # (options, args) = parser.parse_args()
    #
    # #print(options)
    #
    # #if ',' in options.photon_range: options.photon_range = [float(n) for n in options.photon_range.split(',')]
    # #if ',' in options.nsb_range: options.nsb_range = [float(n) for n in options.nsb_range.split(',')]
    # #options.photon_times = [float(n) for n in options.photon_times.split(',')]
    #
    #
    # init_param = {'start_time' :options.photon_times[0], 'end_time': options.photon_times[1], 'sampling_time' : options.photon_times[2], 'nsb_rate' : options.nsb_range[1], 'mean_crosstalk_production' : options.mean_crosstalk,
    #               'n_signal_photon' : options.photon_range[0], 'sig_poisson': options.sig_poisson, 'sigma_e' : options.sigma_e, 'sigma_1': options.sigma_1, 'gain' : options.gain, 'baseline': options.baseline}
    #
    #
    # options.target_segmentation = float(options.target_segmentation)
    # # Create the trace generator
    # tr = Trace_Generator(**init_param)
    # # Create the file
    # filename = options.directory + options.filename
    # print('--|> Creating file for mu = %0.2f [p.e.], XT = %0.2f, dark_rate = %0.2f [MHz]' %(options.photon_range[0], options.mean_crosstalk, options.nsb_range[0]*1E3))
    #
    #
    # createFile( options,tr , filename)
    #
    # print('--|> File created for : ', tr.__dict__['n_signal_photon'], ' [p.e.]')

