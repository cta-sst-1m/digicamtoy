import os.path
from scipy.interpolate import interp1d
import copy
import pandas as pd
import numpy as np

from digicamtoy.utils.pulse_shape import return_interpolant
import digicamtoy
interpolant = return_interpolant()

"""
from numba import jit, int64, float32


@jit((int64(float32)))
def _poisson_crosstalk(crosstalk):
    n_cross_talk = np.random.poisson(crosstalk)
    counter = n_cross_talk

    for i in range(n_cross_talk):
        counter += _poisson_crosstalk(crosstalk)

    return counter

@jit((int64(float32, int64)))
def _generalized_poisson_crosstalk(crosstalk, n_photon):

    counter = 0
    for i in range(n_photon):

        counter += _poisson_crosstalk(crosstalk)

    return counter
"""


class NTraceGenerator:

    def __init__(
        self,
        time_start=0,
        time_end=200,
        time_sampling=4,
        n_pixels=1296,
        nsb_rate=0.6,
        crosstalk=0.08,
        gain_nsb=True,
        n_photon=0,
        poisson=True,
        sigma_e=0.8,
        sigma_1=0.8,
        gain=5.8,
        baseline=200,
        time_signal=20,
        jitter=0,
        pulse_shape_file='/utils/pulse_SST-1M_pixel_0.dat',
        sub_binning=0,
        **kwargs
    ):
        '''
        time_start: scalar
            start time of sampled waveforms (in ns?).

        time_end: scalar
            end time of sampled waveforms in the same unit as time_start.

        time_sampling: scalar
            number of samples per unit of time. or units of time per sample.
            (digicam samples with 250MHz so a sample is 4ns long.)

        sigma_e: scalar or 1-d array of shape (n_pixels, )
            electronics noise (std-dev of normal distribution)
            electronics noise is simulated as white noise.

        jitter: scalar or 1d or 2d array of shape: ???
            I do not know what kind of jitter is meant here, and in what
            unit it might be given.

        n_photon: sclar or 1d or 2d of shape: ???
            the expectation value of the number of photons
            per pixel and per event.
            The actual number of photons per pixel and per event is drawn from
            a poisson distribution, with the expectation value `n_photons`.
            Even negative values are allowed.

        time_signal: scalar or 1d or 2d of shape: ???
            the time (in samples) where the cherencov signal should be simulated.
        '''

        # np.random.seed(seed)

        nsb_rate = np.atleast_1d(nsb_rate)
        nsb_rate = nsb_rate if nsb_rate.shape[0] > 1 else \
            np.array([nsb_rate[0]]*n_pixels)
        crosstalk = np.atleast_1d(crosstalk)
        crosstalk = crosstalk if crosstalk.shape[0] > 1 \
            else np.array([crosstalk[0]]*n_pixels)

        if isinstance(n_photon, int) or isinstance(n_photon, float):

            n_photon = [[n_photon]]*n_pixels
            n_photon = np.atleast_2d(n_photon)

        else:
            n_photon = pd.DataFrame(n_photon).fillna(0).values

        sigma_e = np.atleast_1d(sigma_e)
        sigma_e = sigma_e if sigma_e.shape[0] > 1 else\
            np.array([sigma_e[0]] * n_pixels)
        sigma_1 = np.atleast_1d(sigma_1)
        sigma_1 = sigma_1 if sigma_1.shape[0] > 1 else\
            np.array([sigma_1[0]] * n_pixels)
        gain = np.atleast_1d(gain)
        gain = gain if gain.shape[0] > 1 else \
            np.array([gain[0]] * n_pixels)
        baseline = np.atleast_1d(baseline)
        baseline = baseline if baseline.shape[0] > 1 else \
            np.array([baseline[0]] * n_pixels)

        if isinstance(time_signal, int) or isinstance(time_signal, float):
            time_signal = [[time_signal]] * n_pixels
            time_signal = np.atleast_2d(time_signal)

        else:
            time_signal = pd.DataFrame(time_signal).fillna(0).values

        if isinstance(jitter, int) or isinstance(jitter, float):
            jitter = [[jitter]] * n_pixels
            jitter = np.atleast_2d(jitter)
        else:
            jitter = pd.DataFrame(jitter).fillna(0).values

        self.artificial_backward_time = 40
        # n_samples of the output waveform
        self.n_samples = (time_end - time_start) // time_sampling
        self.time_start = time_start - self.artificial_backward_time
        self.time_end = time_end
        self.time_sampling = time_sampling
        self.nsb_rate = nsb_rate
        self.crosstalk = crosstalk
        self.n_pixels = n_pixels
        self.n_photon = n_photon
        self.time_signal = time_signal
        self.jitter = jitter
        self.poisson = poisson
        directory = os.path.dirname(digicamtoy.__file__)
        self.filename_pulse_shape = directory + pulse_shape_file
        self.gain = gain
        self.sigma_e = sigma_e
        self.gain_nsb = gain_nsb
        self.cell_capacitance = 85. * 1E-15
        self.bias_resistance = 10. * 1E3
        self.gain = gain / (1. + nsb_rate * self.cell_capacitance
                            * self.bias_resistance * 1E9
                            * self.gain_nsb)
        self.baseline = baseline
        self.sigma_1 = sigma_1 / self.gain

        time_steps, amplitudes = np.loadtxt(self.filename_pulse_shape,
                                            unpack=True)
        self.pulse_template = interp1d(time_steps, amplitudes, kind='cubic',
                                       bounds_error=False, fill_value=0.,
                                       assume_sorted=True)
        temp_times = np.arange(self.time_start,
                               self.time_end,
                               self.time_sampling / 100)

        temp_amplitudes = self.pulse_template(temp_times)
        amplitudes = amplitudes / np.max(temp_amplitudes)
        self.tau = np.trapz(temp_amplitudes / np.max(temp_amplitudes),
                            temp_times)
        self.pulse_template = interp1d(time_steps, amplitudes, kind='cubic',
                                       bounds_error=False, fill_value=0.,
                                       assume_sorted=True)

        self.sub_binning = sub_binning
        self.count = -1
        self.reset()

    def __str__(self):

        return ''.join('{}{}'.format(key, val) for key, val in
                       sorted(self.__dict__.items()))

    def __iter__(self):

        return self

    def __next__(self):

        self.next()

        return self

    def get_pulse_shape(self, time, t_0=0, baseline=0, amplitude=1):

        return amplitude * self.pulse_template(time - t_0) + baseline

    def reset(self):

        self.cherenkov_time = np.zeros(self.n_photon.shape)
        self.cherenkov_photon = np.zeros(self.n_photon.shape, dtype=np.int)
        self.nsb_time = np.zeros((self.n_pixels, 1))
        self.nsb_photon = np.zeros((self.n_pixels, 1))
        self.mask = np.zeros(self.nsb_photon.shape)
        self.sampling_bins = np.arange(self.time_start, self.time_end,
                                       self.time_sampling)
        self.adc_count = np.zeros((self.n_pixels, self.sampling_bins.shape[0]))

    def next(self):

        self.reset()
        self.count += 1

        if np.any(self.nsb_rate > 0):
            self.generate_nsb()

        if np.any(self.n_photon > 0):
            self.add_signal_photon()

        if np.any(self.crosstalk > 0):
            self.generate_crosstalk()

        if np.any(self.sigma_1 > 0):
            self.generate_photon_smearing()

        self.compute_analog_signal()

        if np.any(self.sigma_e > 0):

            self.generate_electronic_smearing()

        self.convert_to_digital()

    def add_signal_photon(self):
        '''
        set two internal variables:
            self.cherenkov_time,
            self.cherenkov_photon

        * calculate arrival time of cherenkov-photon-bundle: time_signal + jitter
        * calculate size of cherenkov-photon-bundle: from poisson distribution.
        '''

        jitter = copy.copy(self.jitter)
        mask = jitter <= 0
        jitter[mask] = 1
        jitter = np.random.uniform(0, jitter)
        jitter[mask] = 0
        self.cherenkov_time = self.time_signal + jitter
        self.cherenkov_photon = (
            np.random.poisson(lam=self.n_photon)
            if self.poisson
            else self.n_photon
        )
        self.cherenkov_photon[self.n_photon <= 0] = 0

    def generate_nsb(self):
        mean_nsb_pe = (self.time_end - self.time_start) * self.nsb_rate
        photon_number = np.random.poisson(lam=mean_nsb_pe)
        max_photon = np.max(photon_number)

        if self.sub_binning <= 0:

            self.nsb_time = np.random.uniform(size=(self.n_pixels, max_photon))
            self.nsb_time *= (self.time_end - self.time_start)
            self.nsb_time += self.time_start

            self.nsb_photon = np.ones(self.nsb_time.shape, dtype=int)

        else:

            sub_bins = np.arange(
                self.time_start,
                self.time_end,
                self.sub_binning
            )
            hist = np.random.randint(
                0,
                sub_bins.shape[-1],
                size=(self.n_pixels, max_photon)
            )

        self.mask = np.arange(max_photon)
        self.mask = np.tile(self.mask, (self.n_pixels, 1))
        self.mask = (self.mask < photon_number[..., np.newaxis])
        self.mask = self.mask.astype(int)

    def _poisson_crosstalk(self, crosstalk):

        n_cross_talk = np.random.poisson(crosstalk)
        counter = n_cross_talk

        for i in range(n_cross_talk):
            counter += self._poisson_crosstalk(crosstalk)

        return counter

    def _poisson_crosstalk_2(self, crosstalk, counter=0):

        n_cross_talk = np.random.poisson(crosstalk)

        if n_cross_talk == 0:

            return counter

        else:

            counter += n_cross_talk
            i = 0

            while i < n_cross_talk:
                counter += self._poisson_crosstalk_2(crosstalk, counter)
                i += 1
            return counter

    def generate_crosstalk(self):

        nsb_crosstalk = np.zeros(self.nsb_photon.shape, dtype=int)
        cherenkov_crosstalk = np.zeros(self.cherenkov_photon.shape, dtype=int)

        for pixel in range(self.n_pixels):

            for i in range(self.nsb_photon.shape[-1]):

                nsb_crosstalk[pixel, i] = self._poisson_crosstalk(
                    self.crosstalk[pixel])

            for j in range(self.cherenkov_photon.shape[-1]):

                for k in range(self.cherenkov_photon[pixel, j]):

                    cherenkov_crosstalk[pixel, j] += self._poisson_crosstalk(
                        self.crosstalk[pixel])

        self.nsb_photon += nsb_crosstalk
        self.cherenkov_photon += cherenkov_crosstalk

    def generate_photon_smearing(self):

        nsb_smearing = np.sqrt(self.nsb_photon) * self.sigma_1[:, np.newaxis]
        cherenkov_smearing = (
            np.sqrt(self.cherenkov_photon) *
            self.sigma_1[:, np.newaxis]
        )

        mask = nsb_smearing <= 0
        nsb_smearing[mask] = 1
        nsb_smearing = np.random.normal(0, nsb_smearing)
        nsb_smearing[mask] = 0

        mask = cherenkov_smearing <= 0
        cherenkov_smearing[mask] = 1
        cherenkov_smearing = np.random.normal(0, cherenkov_smearing)
        cherenkov_smearing[mask] = 0

        self.nsb_photon = self.nsb_photon.astype(np.float32)
        self.cherenkov_photon = self.cherenkov_photon.astype(np.float32)
        self.nsb_photon += nsb_smearing
        self.cherenkov_photon += cherenkov_smearing

    def compute_analog_signal(self):
        '''
        self.sampling_bins: 1d, (n_samples,)
        self.nsb_time: 2d, (n_pixels, some_number)
        times: 3d, (n_pixels, some_number, n_samples)
        self.nsb_photon: 2d, (n_pixels, 1)
        self.mask: 1d, (maxphoton, )

        I think some_number and maxphoton might be the same

            * sample noise photons onto the timeline
            * sample cherencov photons onto the timeline
            * multiply with gain
        '''

        times = self.sampling_bins - self.nsb_time[..., np.newaxis]
        self.adc_count = (
            self.get_pulse_shape(times) *
            (self.nsb_photon * self.mask)[..., np.newaxis]
        )
        self.adc_count = np.sum(self.adc_count, axis=1)
        times = self.sampling_bins - self.cherenkov_time[..., np.newaxis]
        temp = (
            self.get_pulse_shape(times) *
            self.cherenkov_photon[..., np.newaxis]
        )
        temp = np.sum(temp, axis=1)
        self.adc_count += temp
        self.adc_count = self.adc_count * self.gain[..., np.newaxis]

    def generate_electronic_smearing(self):
        '''
        self.sigma_e is 1d of shape: (n_pixel, )

        * make sure self.sigma_e is positive
        * convert the scalar self.sigma_e into an array of identical
            values with shape (n_pixel, n_samples)
        * use this as the `scale` parameter of np.random.normal
        * add a white noise to self.adc_count
        '''

        smearing_spread = np.sqrt(self.sigma_e ** 2)
        smearing_spread = np.tile(smearing_spread,
                                  (self.sampling_bins.shape[-1], 1)).T
        smearing = np.random.normal(0., smearing_spread)

        self.adc_count += smearing

    def convert_to_digital(self):
        '''
        * remove some artificial samples at the beginning
        * add baseline
        * and only then: digitize
        '''

        n_bins_to_remove = self.artificial_backward_time // self.time_sampling
        indices_to_remove = range(0, n_bins_to_remove, 1)
        self.adc_count = np.delete(self.adc_count, indices_to_remove, axis=1)
        self.sampling_bins = np.delete(self.sampling_bins, indices_to_remove)
        self.adc_count = (self.adc_count + self.baseline[..., np.newaxis])
        self.adc_count = self.adc_count.round()
        self.adc_count = self.adc_count.astype(np.uint)


'''
class TraceGenerator:

    def __init__(self, start_time=-100., end_time=100., sampling_time=4., nsb_rate=660 * 1E6 * 1E-9,
                 mean_crosstalk_production=0.08, debug=False, gain_nsb_dependency=False, n_signal_photon=0.,
                 sig_poisson=True, sigma_e=0.8, sigma_1=0.8, gain=5.6, baseline=2010., time_signal=0, jitter_signal=0,
                 pulse_shape_spline=None, *args, **kwargs):

        ## Initialize class attributs

        self.debug = debug  # boolean for debuging
        self.debug_counter = 0  # debug counter avoiding excessive debug calls
        self.artificial_backward_time = 80.  # artificial backward time to avoid boundary effects
        self.start_time = start_time - self.artificial_backward_time  # start time of the trace
        self.end_time = end_time  # end time of the trace
        self.sampling_time = sampling_time  # FADC sampling time
        self.nsb_rate = nsb_rate  # night sky background or dark count rate per SiPM
        self.mean_crosstalk_production = mean_crosstalk_production  # mean number of crosstalk produced per fired cell
        self.sampling_bins = np.arange(self.start_time, self.end_time,
                                       self.sampling_time)  # FADC sampling times
        self.adc_count = np.zeros(len(self.sampling_bins))  # FADC values
        self.n_signal_photon = n_signal_photon  # number of Cherenkov photons
        self.time_signal = time_signal
        self.cherenkov_time = np.inf
        self.jitter_signal = jitter_signal
        self.sig_poisson = sig_poisson  # is the number of signal photons distributed as a poisson?
        dir = os.path.dirname(digicamtoy.__file__)
        self.filename_pulse_shape = dir + '/utils/pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file
        self.pe_to_adc = gain
        self.sigma_e = sigma_e / gain  # electronic spread in analog to digital conversion
        self.baseline = int(baseline)
        self.cell_capacitance = 85. * 1E-15  # Farad
        self.bias_resistance = 10. * 1E3  # Ohm
        self.gain_nsb_dependency = gain_nsb_dependency
        self.gain = 1. / (
            1. + nsb_rate * self.cell_capacitance * self.bias_resistance * 1E9 * self.gain_nsb_dependency)  # gain taking into account nsb dependency
        self.sigma_1 = sigma_1 / (self.gain * self.pe_to_adc)  # spread due to charge resolution in photocounting

        time_steps, amplitudes = np.loadtxt(self.filename_pulse_shape, unpack=True, skiprows=1)
        self.time_steps_pulse_shape = time_steps
        amplitudes = amplitudes / min(amplitudes)
        self.amplitudes_pulse_shape = amplitudes

        # TODO change seed by passing the np.random.RandomState() instead

        self.plot_title = ''  # 'NSB = ' + str(self.nsb_rate * 1E3) + ' [MHz], $\mu_{XT}$ = ' + str(
        # self.mean_crosstalk_production) + ' p.e.'
        self.photon_arrival_time = np.zeros(1)
        self.photon = np.zeros(1)

        # Ploting results

    def compute_gain_drop(self, cell_capacitance, bias_resistance, nsb_rate):
        return

    def __str__(self):

        return ''.join('{}{}'.format(key, val) for key, val in sorted(self.__dict__.items()))

    def __iter__(self):
        return self

    def __next__(self):

        return self.next()

    def fast_config(self, n_signal, nsb_rate):
        self.nsb_rate = nsb_rate
        self.n_signal_photon = n_signal

    def next(self):
        self.photon_arrival_time = np.zeros(1)
        self.photon = np.zeros(1)
        self.sampling_bins = np.arange(self.start_time, self.end_time + self.sampling_time,
                                       self.sampling_time)  # FADC sampling times
        self.adc_count = np.zeros(len(self.sampling_bins))  # FADC values

        ## Compute the FADC signal
        if (self.n_signal_photon > 0):
            self.add_signal_photon()

        if self.nsb_rate > 0:
            self.generate_nsb()

        if self.mean_crosstalk_production > 0:
            self.generate_crosstalk()

        if self.sigma_1 > 0:
            self.generate_photon_smearing()

        self.compute_analog_signal()
        self.convert_to_digital()
        # return self.n_signal_photon,self.nsb_rate,self.adc_count

        axis = None
        if (self.debug):
            # plt.figure()
            # plt.title(self.plot_title)
            axis = plt.step(self.sampling_bins, self.adc_count,
                            label='%d [MHz]' % (self.nsb_rate * 1E3))  # , color='b')
            # plt.ylabel(r'$ADC count$')
            # plt.xlabel(r'$t$ [ns]')
            # plt.xlim(np.array([self.time_start + self.artificial_backward_time, self.time_end]))
            # ~ plt.ylim([0, 150])
            # plt.legend()

            # plt.figure()
            # plt.title(self.plot_title)
            # plt.hist(self.adc_count, bins=range(min(self.adc_count), max(self.adc_count) + 2, 1), facecolor='green',
            #         alpha=0.5, align='left', histtype='step')
            # plt.xlabel(r'$ADC count$')
            # plt.ylabel(r'count')

        return axis

    def interpolated_pulseshape(self, time):

        return interpolant(time)

    def get_adc_count(self):

        return self.adc_count

    def add_signal_photon(self):

        if self.jitter_signal > 0:

            jitter = np.random.normal(0, self.jitter_signal)

        else:

            jitter = 0.

        self.photon_arrival_time[
            0] = self.time_signal + jitter  # -1.54 #np.random.uniform(-self.time_sampling, self.time_sampling) # + np.random.normal(0, self.time_sampling/100., size=1)
        self.cherenkov_time = self.photon_arrival_time[0]
        self.photon[0] = np.random.poisson(self.n_signal_photon) if self.sig_poisson else self.n_signal_photon

    def generate_nsb(self):
        """ create NSB or dark count photons
        1) the number of photon is randomly created according to NSB level
        2) photons are distrbuted randomly along the time axis
        """
        mean_photon_number = np.random.poisson((self.end_time - self.start_time) * self.nsb_rate)

        # print(mean_photon_number)

        # ~ if (mean_photon_number>0):

        temp = np.random.uniform(size=mean_photon_number) * (
            self.end_time - self.start_time) + self.start_time
        self.photon_arrival_time = np.append(self.photon_arrival_time, temp)
        self.photon = np.append(self.photon, np.ones(len(self.photon_arrival_time)))

        indices_sorted_time = np.argsort(self.photon_arrival_time)
        self.photon_arrival_time = self.photon_arrival_time[indices_sorted_time]
        self.photon = self.photon[indices_sorted_time]

        if (self.debug):

            label = ' '

            if (n_cherenkov_photon > 0):

                label = 'NSB + Cherenkov, $t_{signal} =$ ' + str(self.cherenkov_time) + ' [ns]'

            else:

                label = 'NSB'

            plt.figure()
            plt.title(self.plot_title)
            plt.bar(self.photon_arrival_time, self.photon, label=label, color='k')
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'$N(t)$')
            plt.xlim([self.start_time, self.end_time])
            plt.legend()

        return

    def _poisson_crosstalk(self, mean_crosstalk_production):

        N_cross_talk = np.random.poisson(mean_crosstalk_production)
        counter = N_cross_talk
        for i in range(N_cross_talk):
            counter += self._poisson_crosstalk(mean_crosstalk_production)

        return counter

    def generate_crosstalk(self):
        """ create crosstalk for each existing photon
        1) the number of crosstalk photons is randomly generated
        2) crosstalk photons are added to the photon list
        """
        crosstalk_photons = np.zeros(len(self.photon))
        for i in range(len(self.photon)):
            for j in range(int(self.photon[i])):
                # crosstalk_photons[i] += np.random.poisson(self.mean_crosstalk_production)
                crosstalk_photons[i] += self._poisson_crosstalk(self.mean_crosstalk_production)

        self.photon += crosstalk_photons

        if (self.debug):

            label = ' '

            if (n_cherenkov_photon > 0):

                label = 'NSB + XT + Cherenkov, $t_{signal} =$ ' + str(self.cherenkov_time) + ' [ns]'

            else:

                label = 'NSB + XT'

            plt.figure()
            plt.title(self.plot_title)
            plt.bar(self.photon_arrival_time, self.photon, label=label, color='r')
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'$N(t)$')
            plt.xlim(np.array([self.start_time, self.end_time]))
            plt.legend()

        return

    def generate_photon_smearing(self):
        """ generate smearing due to charge resolution
        1) create a set of gaussian(mean=0, std=N_photon*sigma_1) values (one for each photon)
        2) add the smearing to the photon
        """

        smearing_spread = np.sqrt(self.photon * self.sigma_1 ** 2)

        smearing = np.zeros(len(smearing_spread))

        for i in range(len(smearing_spread)):

            if smearing_spread[i] > 0:

                smearing[i] = np.random.normal(0., smearing_spread[i])

            else:

                smearing[i] = 0.

        self.photon += smearing

        if (self.debug):

            label = ' '

            if (n_cherenkov_photon > 0):

                label = 'NSB + XT + Smearing + Cherenkov, $t_{signal} =$ ' + str(self.cherenkov_time) + ' [ns]'

            else:

                label = 'NSB + XT + Smearing'

            plt.figure()
            plt.title(self.plot_title)
            plt.bar(self.photon_arrival_time, self.photon, label=label, color='b')
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'$N(t)$')
            plt.xlim(np.array([self.start_time, self.end_time]))
            plt.legend()

        return

    def generate_electronic_smearing(self):
        """ generate smearing electronic due to the reading of FADC
        1) create a set of gaussian(mean=0, std=sigma_e) values (one for each photon)
        2) add the smearing to the photon
        """

        smearing_spread = np.sqrt(self.sigma_e ** 2)
        smearing = np.random.normal(0., smearing_spread, size=len(self.sampling_bins))

        self.adc_count += smearing

        return

    def return_pulseshape(self, time):
        """ return an interpolated value of the pulse shape taken from a template file
        1) read the template file
        2) normalize amplitudes such that it is positive and always smaller than 1
        3) return spline value at given time
        """

        if (self.debug and self.debug_counter == 0):
            self.debug_counter += 1
            delta_t = 0.1
            debug_time = np.arange(self.start_time + self.artificial_backward_time, self.end_time + delta_t, delta_t)
            # ~ print debug_time
            # print np.array(self.interpolated_pulseshape(debug_time))

            plt.figure()
            plt.plot(debug_time, self.interpolated_pulseshape(debug_time), label='p.e. event', color='r')
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'amplitude [a.u.]')
            plt.xlim(np.array([self.start_time, self.end_time]))
            plt.legend()

        return self.interpolated_pulseshape(time)

    def compute_analog_signal(self):

        ## Compute by suming pulse shape

        # self.generate_electronic_smearing() # Carefull !!

        for i in range(len(self.photon)):
            self.adc_count += self.return_pulseshape(self.sampling_bins - self.photon_arrival_time[i]) * self.photon[i]

        self.adc_count = self.adc_count * self.gain

        if (self.debug):

            delta_t = 0.01
            debug_time = np.arange(self.start_time + self.artificial_backward_time, self.end_time + delta_t, delta_t)
            debug_analog = np.zeros(len(debug_time))

            for i in range(len(self.photon)):
                debug_analog += self.return_pulseshape(debug_time - self.photon_arrival_time[i]) * self.photon[i]

            plt.figure()
            plt.title(self.plot_title)
            plt.plot(debug_time, debug_analog, label='analog signal', color='r', )
            plt.xlabel(r'$t$ [ns]')
            plt.ylabel(r'$N(t)$')
            plt.xlim(np.array([self.start_time, self.end_time]))
            plt.legend()

        return

    def convert_to_digital(self):

        indices_to_remove = range(0, int(self.artificial_backward_time / self.sampling_time), 1)
        self.adc_count = np.delete(self.adc_count, indices_to_remove)
        self.sampling_bins = np.delete(self.sampling_bins, indices_to_remove)
        self.photon = self.photon[self.photon_arrival_time >= (self.start_time + self.artificial_backward_time)]
        self.photon_arrival_time = self.photon_arrival_time[
            self.photon_arrival_time >= (self.start_time + self.artificial_backward_time)]

        self.generate_electronic_smearing()  # TODO check for correlations ??? with gain ?
        self.adc_count = self.adc_count * self.pe_to_adc
        # self.generate_electronic_smearing() # Carefull !!
        self.adc_count = self.adc_count.round().astype(int) + self.baseline

    def save(self, mc_number, path='./'):

        filename = path + 'TRACE' + str(mc_number).zfill(5) + '.p'

        if os.path.isfile(filename):
            print('file named : ' + filename + ' already exists')
            sys.exit("Error message")

        with open(filename, 'wb') as handle:
            pickle.dump((self.adc_count, self.end_time - self.start_time - self.artificial_backward_time,
                         self.sampling_time, self.nsb_rate, self.mean_crosstalk_production, self.n_cherenkov_photon),
                        handle)

        return

'''
