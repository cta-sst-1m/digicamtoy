import digicamtoy
from digicamtoy.utils.analytical import true_pde_drop, true_xt_drop, \
    true_gain_drop
import os.path
from scipy.interpolate import interp1d
from scipy import linalg
from scipy.stats import multivariate_normal
import copy
import pandas as pd
import numpy as np

from digicamtoy.utils.pulse_shape import return_interpolant
from digicamtoy.utils.analytical import true_gain_drop, true_pde_drop, true_xt_drop
import digicamtoy
interpolant = return_interpolant()


class NTraceGenerator:

    def __init__(self,
                 time_start=0, time_end=200, time_sampling=4, n_pixels=1296,
                 nsb_rate=0.6, crosstalk=0.08, gain_nsb=True, n_photon=0,
                 poisson=True, sigma_e=0.8, sigma_1=0.8, gain=5.8,
                 baseline=200, time_signal=20, jitter=0,
                 pulse_shape_file='/utils/pulse_SST-1M_pixel_0.dat',
                 sub_binning=0,
                 n_events=None, voltage_drop=False,
                 verbose=True,
                 **kwargs  # TODO not allow **kwargs (change in produce_data)
    ):
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

        self.n_pixels = n_pixels
        self.n_photon = n_photon
        self.time_signal = time_signal
        self.jitter = jitter
        self.poisson = poisson
        directory = os.path.dirname(digicamtoy.__file__)
        self.filename_pulse_shape = os.path.join(directory, pulse_shape_file)
        self.gain = gain
        self.sigma_e = sigma_e
        self.gain_nsb = gain_nsb
        self.voltage_drop = voltage_drop

        if voltage_drop:

            self.nsb_rate = nsb_rate * true_pde_drop(nsb_rate * 1E9)
            self.crosstalk = crosstalk * true_xt_drop(nsb_rate * 1E9)
            self.gain = gain * true_gain_drop(nsb_rate * 1E9)
            self.n_photon = self.n_photon * true_pde_drop(nsb_rate * 1E9)

        elif gain_nsb:
            self.gain = gain / (1. + nsb_rate / self.gain_nsb)
            self.crosstalk = crosstalk
            self.nsb_rate = nsb_rate

        else:

            self.gain = gain
            self.nsb_rate = nsb_rate
            self.crosstalk = crosstalk

        if verbose:

            print('Set NSB rate : {} [GHz] \tTrue NSB rate : {} [GHz]'
                  ''.format(nsb_rate.mean(), self.nsb_rate.mean()))
            print('Set XT : {} [p.e.] \tTrue XT : {} [p.e.]'
                  ''.format(crosstalk.mean(), self.crosstalk.mean()))
            print('Set Gain : {} [LSB/p.e.] \tTrue Gain : {} [LSB/p.e.]'
                  ''.format(gain.mean(), self.gain.mean()))
        self.sigma_1 = sigma_1 / self.gain
        self.baseline = baseline

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

        self.true_baseline = self.baseline + self.gain * self.nsb_rate * (
                1 / (1 - self.crosstalk)) * self.tau

        if verbose:
            print('Set Baseline : {} [LSB]'
                  '\tTrue Baseline : {} [LSB]'
                  '\tTrue Baseline Shift : {} [LSB]'
                  ''.format(self.baseline.mean(),
                            self.true_baseline.mean(),
                            self.true_baseline.mean() - self.baseline.mean()))

        self.sub_binning = sub_binning
        self.count = -1
        self.n_events = n_events
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
        self.nsb_photon = np.zeros((self.n_pixels, 1))
        self.nsb_time = np.zeros((self.n_pixels, 1))
        self.mask = np.zeros(self.nsb_photon.shape)
        self.sampling_bins = np.arange(self.time_start, self.time_end,
                                       self.time_sampling)
        self.adc_count = np.zeros((self.n_pixels, self.sampling_bins.shape[0]))

    def next(self):

        if self.n_events is not None and self.count >= self.n_events:
            raise StopIteration
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

        jitter = copy.copy(self.jitter)
        mask = jitter <= 0
        jitter[mask] = 1
        jitter = np.random.uniform(-jitter/2, jitter/2)
        jitter[mask] = 0
        self.cherenkov_time = self.time_signal + jitter
        self.cherenkov_photon = np.random.poisson(lam=self.n_photon) if \
            self.poisson else self.n_photon.astype(np.int)
        self.cherenkov_photon[self.n_photon <= 0] = 0

    def generate_nsb(self):

        if self.sub_binning <= 0:

            mean_nsb_pe = (self.time_end - self.time_start) * self.nsb_rate
            photon_number = np.random.poisson(lam=mean_nsb_pe)
            max_photon = np.max(photon_number)

            self.nsb_time = np.random.uniform(size=(self.n_pixels, max_photon))
            self.nsb_time *= (self.time_end - self.time_start)
            self.nsb_time += self.time_start

            self.nsb_photon = np.ones(self.nsb_time.shape, dtype=int)

            self.mask = np.arange(max_photon)
            self.mask = np.tile(self.mask, (self.n_pixels, 1))
            self.mask = (self.mask < photon_number[..., np.newaxis])
            self.mask = self.mask.astype(int)

        else:

            # if self.count == 0:

            self.nsb_time = np.tile(self.sampling_bins, (self.n_pixels, 1))
            n_nsb = self.nsb_rate * self.time_sampling
            n_nsb = np.tile(n_nsb, (len(self.sampling_bins), 1)).T
            self.nsb_photon = np.random.poisson(lam=n_nsb)
            self.mask = self.nsb_photon > 0

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

                if self.sub_binning > 0:

                    for _ in range(self.nsb_photon[pixel, i]):

                        nsb_crosstalk[pixel, i] += self._poisson_crosstalk(
                            self.crosstalk[pixel]
                        )

                else:

                    nsb_crosstalk[pixel, i] = self._poisson_crosstalk(
                        self.crosstalk[pixel])

            for j in range(self.cherenkov_photon.shape[-1]):

                for _ in range(self.cherenkov_photon[pixel, j]):

                    cherenkov_crosstalk[pixel, j] += self._poisson_crosstalk(
                        self.crosstalk[pixel])

        self.nsb_photon += nsb_crosstalk
        self.cherenkov_photon += cherenkov_crosstalk

    def generate_photon_smearing(self):

        nsb_smearing = np.sqrt(self.nsb_photon) * self.sigma_1[:, np.newaxis]
        cherenkov_smearing = np.sqrt(self.cherenkov_photon) * \
                             self.sigma_1[:, np.newaxis]

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

        times = self.sampling_bins - self.nsb_time[..., np.newaxis]
        self.adc_count = self.get_pulse_shape(times) * \
                         (self.nsb_photon * self.mask)[..., np.newaxis]

        self.adc_count = np.sum(self.adc_count, axis=1)
        times = self.sampling_bins - self.cherenkov_time[..., np.newaxis]
        temp = self.get_pulse_shape(times) * \
               self.cherenkov_photon[..., np.newaxis]
        temp = np.sum(temp, axis=1)
        self.adc_count += temp
        self.adc_count = self.adc_count * self.gain[..., np.newaxis]

    def generate_electronic_smearing(self):

        smearing_spread = np.sqrt(self.sigma_e ** 2)
        smearing_spread = np.tile(smearing_spread,
                                  (self.sampling_bins.shape[-1], 1)).T
        smearing = np.random.normal(0., smearing_spread)

        self.adc_count += smearing

    def convert_to_digital(self):

        n_bins_to_remove = self.artificial_backward_time // self.time_sampling
        indices_to_remove = range(0, n_bins_to_remove, 1)
        self.adc_count = np.delete(self.adc_count, indices_to_remove, axis=1)
        self.sampling_bins = np.delete(self.sampling_bins, indices_to_remove)
        self.adc_count = (self.adc_count + self.baseline[..., np.newaxis])
        self.adc_count = self.adc_count.round()
        self.adc_count = self.adc_count.astype(np.uint)