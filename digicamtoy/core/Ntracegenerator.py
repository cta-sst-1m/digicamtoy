import os.path
import numpy as np
from scipy.interpolate import interp1d
from digicamtoy.utils.pulse_shape import return_interpolant
import digicamtoy

interpolant = return_interpolant()


class NTraceGenerator:

    def __init__(self, time_start=0, time_end=200, time_sampling=4, n_pixels=1296, nsb_rate=np.ones(1296) * 0.6,
                 crosstalk=np.ones(1296) * 0.08, gain_nsb_dependency=True, n_photon=np.zeros(1296), poisson=True,
                 sigma_e=np.ones(1296) * 0.8, sigma_1=np.ones(1296) * 0.8, gain=np.ones(1296) * 5.8,
                 baseline=np.ones(1296) * 200., time_signal=np.ones(1296) * 20, jitter=np.zeros(1296),
                 pulse_shape_file='/utils/pulse_SST-1M_AfterPreampLowGain.dat'):

        self.artificial_backward_time = 40
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
        self.gain_nsb_dependency = gain_nsb_dependency
        self.cell_capacitance = 85. * 1E-15
        self.bias_resistance = 10. * 1E3
        self.gain = gain / (1. + nsb_rate * self.cell_capacitance * self.bias_resistance * 1E9 * self.gain_nsb_dependency)
        self.baseline = baseline.astype(int)
        self.sigma_1 = sigma_1 / self.gain

        time_steps, amplitudes = np.loadtxt(self.filename_pulse_shape, unpack=True, skiprows=1)
        amplitudes = amplitudes / min(amplitudes)
        self.pulse_template = interp1d(time_steps, amplitudes, kind='cubic', bounds_error=False, fill_value=0., assume_sorted=True)


        # TODO change seed by passing the np.random.RandomState() instead

        self.count = -1
        self.reset()

    def __str__(self):

        return ''.join('{}{}'.format(key, val) for key, val in sorted(self.__dict__.items()))

    def __iter__(self):
        return self

    def __next__(self):

        self.next()
        return self

    def get_pulse_shape(self, time, t_0=0, baseline=0, amplitude=1):

        return amplitude * self.pulse_template(time - t_0) + baseline

    def reset(self):

        self.cherenkov_time = np.zeros(self.n_pixels)
        self.cherenkov_photon = np.zeros(self.n_pixels, dtype=np.int)
        self.nsb_time = np.zeros((self.n_pixels, 1))
        self.nsb_photon = np.zeros((self.n_pixels, 1))
        self.mask = np.zeros(self.nsb_photon.shape)
        self.sampling_bins = np.arange(self.time_start, self.time_end, self.time_sampling)
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

        jitter = self.jitter
        mask = jitter <= 0
        jitter[mask] = 1
        jitter = np.random.uniform(0, jitter)
        jitter[mask] = 0
        self.cherenkov_time = self.time_signal + jitter
        self.cherenkov_photon = np.random.poisson(lam=self.n_photon) if self.poisson else self.n_photon
        self.cherenkov_photon[self.n_photon <= 0] = 0

    def generate_nsb(self):

        photon_number = np.random.poisson(lam=(self.time_end - self.time_start) * self.nsb_rate)
        max_photon = np.max(photon_number)
        self.nsb_time = np.random.uniform(size=(self.n_pixels, max_photon)) * (self.time_end - self.time_start) + self.time_start
        self.mask = np.arange(max_photon)
        self.mask = np.tile(self.mask, (self.n_pixels, 1))
        self.mask = (self.mask < photon_number[..., np.newaxis])
        self.mask = self.mask.astype(int)
        self.nsb_photon = np.ones(self.nsb_time.shape, dtype=int)

    def _poisson_crosstalk(self, mean_crosstalk_production):

        n_cross_talk = np.random.poisson(mean_crosstalk_production)
        counter = n_cross_talk

        for i in range(n_cross_talk):

            counter += self._poisson_crosstalk(mean_crosstalk_production)

        return counter

    def generate_crosstalk(self):

        nsb_crosstalk = np.zeros(self.nsb_photon.shape, dtype=int)
        cherenkov_crosstalk = np.zeros(self.cherenkov_photon.shape, dtype=int)

        for pixel in range(self.n_pixels):

            for i in range(self.nsb_photon.shape[-1]):

                nsb_crosstalk[pixel, i] = self._poisson_crosstalk(self.crosstalk[pixel])

            for ii in range(self.cherenkov_photon[pixel]):

                cherenkov_crosstalk[pixel] += self._poisson_crosstalk(self.crosstalk[pixel])

        self.nsb_photon += nsb_crosstalk
        self.cherenkov_photon += cherenkov_crosstalk

    def generate_photon_smearing(self):

        nsb_smearing = np.sqrt(self.nsb_photon) * self.sigma_1[:, np.newaxis]
        cherenkov_smearing = np.sqrt(self.cherenkov_photon) * self.sigma_1

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
        self.adc_count = self.get_pulse_shape(times) * (self.nsb_photon * self.mask)[..., np.newaxis]
        self.adc_count = np.sum(self.adc_count, axis=1)
        times = self.sampling_bins - self.cherenkov_time[..., np.newaxis]
        temp = self.get_pulse_shape(times) * self.cherenkov_photon[..., np.newaxis]
        self.adc_count = self.adc_count + temp
        self.adc_count = self.adc_count * self.gain[..., np.newaxis]

    def generate_electronic_smearing(self):

        smearing_spread = np.sqrt(self.sigma_e ** 2)
        smearing_spread = np.tile(smearing_spread, (self.sampling_bins.shape[-1], 1)).T
        smearing = np.random.normal(0., smearing_spread)

        self.adc_count += smearing

    def convert_to_digital(self):

        indices_to_remove = range(0, self.artificial_backward_time // self.time_sampling, 1)
        self.adc_count = np.delete(self.adc_count, indices_to_remove, axis=1)
        self.sampling_bins = np.delete(self.sampling_bins, indices_to_remove)
        self.adc_count = (self.adc_count + self.baseline[..., np.newaxis]).round().astype(np.uint)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    toy_iterator = NTraceGenerator(n_photon=np.ones(1296) * 0, nsb_rate=np.ones(1296) * 0.003)
    plt.figure()
    for i, toy in enumerate(toy_iterator):

        plt.step(toy.sampling_bins, toy.adc_count[0])
        plt.step(toy.sampling_bins, toy.adc_count[1295])
        print(toy.count)
        plt.show()
