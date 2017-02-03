import numpy as np
import h5py
from optparse import OptionParser
from trace_generator import Trace_Generator

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

    data_group = f.create_group('data')
    param_group = f.create_group('param')

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




if __name__ == '__main__':


    # Job configuration
    parser = OptionParser()
    parser.add_option("-n", "--evt_max", dest="evt_max",
                      help="maximal number of events", default=10000, type=int)

    parser.add_option("--batch_max", dest="batch_max",
                      help="maximal number of events for batch in memory", default=100, type=int)

    parser.add_option("-d", "--directory", dest="directory",
                      help="Output directory", default="./")

    parser.add_option("-f", "--filename", dest="filename",
                      help="Output file name", default="test")

    parser.add_option("-p", "--photon_range", dest="photon_range",
                      help="range of signal photons", nargs=3, type=float, default=[0., 40., 1.])

    parser.add_option("-b", "--nsb_range", dest="nsb_range",
                      help="range of NSB", default=[0.,200., 20.], nargs=3, type=float)

    parser.add_option("-x", "--crosstalk", dest="mean_crosstalk",
                      help="Mean number of crosstalk produced per discharged microcell", default=0.06, type=float)

    parser.add_option("-g", "--gain", dest="gain",
                      help="FADC gain in ADC/p.e.", default=5.6)

    parser.add_option("--sigma_e", dest="sigma_e",
                      help="electronic noise in p.e.", default=0.86/5.6)

    parser.add_option("--sigma_1", dest="sigma_1",
                      help="SiPM gain smearing in p.e.", default=0.48/5.6)

    parser.add_option("--photon_times", dest="photon_times",
                      help="arrival time range", default=[-50, 50, 4], nargs=3, type=float)

    parser.add_option("--baseline", dest="baseline",
                     help="user set baseline", default=2010., type=float)

    parser.add_option("--target_segmentation", dest="target_segmentation",
                      help="arrival time range", default=4, type=float)

    parser.add_option("--poisson_signal", dest="sig_poisson",
                      help="if 1 the signal is generated according to a Poisson distribution", default=False, type=int)

    #print(parser.rargs)
    #print(parser.nargs)
    #print(parser)
    (options, args) = parser.parse_args()

    #print(options)

    #if ',' in options.photon_range: options.photon_range = [float(n) for n in options.photon_range.split(',')]
    #if ',' in options.nsb_range: options.nsb_range = [float(n) for n in options.nsb_range.split(',')]
    #options.photon_times = [float(n) for n in options.photon_times.split(',')]


    init_param = {'start_time' :options.photon_times[0], 'end_time': options.photon_times[1], 'sampling_time' : options.photon_times[2], 'nsb_rate' : options.nsb_range[1], 'mean_crosstalk_production' : options.mean_crosstalk,
                  'n_signal_photon' : options.photon_range[0], 'sig_poisson': options.sig_poisson, 'sigma_e' : options.sigma_e, 'sigma_1': options.sigma_1, 'gain' : options.gain, 'baseline': options.baseline}


    options.target_segmentation = float(options.target_segmentation)
    # Create the trace generator
    tr = Trace_Generator(**init_param)
    # Create the file
    filename = options.directory + options.filename
    print('--|> Creating file for mu = %0.2f [p.e.], XT = %0.2f, dark_rate = %0.2f [MHz]' %(options.photon_range[0], options.mean_crosstalk, options.nsb_range[0]*1E3))


    createFile( options,tr , filename)

    print('--|> File created for : ', tr.__dict__['n_signal_photon'], ' [p.e.]')

