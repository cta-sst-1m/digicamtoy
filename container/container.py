import numpy as np
import pickle
import datetime

class Calibration_Container():

    """
    a container with all calibration parameters for SST-1M camera
    each field contains a dict() of np.array
    """

    def __init__(self, filename=None):

        if filename is not None:

            self._load(filename)

        else:

            self.pixel_id = [i for i in range(1296)]
            self.n_pixels = len(self.pixel_id)

            ### SiPMs

            self.gain = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.electronic_noise = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.gain_smearing = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.crosstalk = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.baseline = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.mean_temperature = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.dark_count_rate = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}

            ### LEDs

            self.ac_led = {'value' : [[None]*4]*self.n_pixels, 'error': [[None]*4]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.dc_led = {'value' : [[None]*2]*self.n_pixels, 'error': [[None]*2]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}



    def update(self, field, indices, value, error=None):

        class_attribute = getattr(self, field)

        for i, index in enumerate(indices):

            class_attribute['value'][index] = value[i]

            if error is not None:

                class_attribute['error'][index] = error[i]

            class_attribute['time_stamp'][index] = datetime.datetime.now()



    def save(self, filename):

        with open(filename, 'wb') as output:

            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def _load(self, filename):

        with open(filename, 'rb') as output:

            tmp_dict = pickle.load(output)
            self.__dict__.update(tmp_dict)

    def initialize_standard_camera(self, gain=5.6, electronic_noise=0.8, gain_smearing=0.8, crosstalk=0.07, baseline=500, mean_temperature=19, dark_count_rate=3.):

        self.update('gain', self.pixel_id, np.random.normal(gain, 0.1, size=self.n_pixels))
        self.update('electronic_noise', self.pixel_id, np.random.normal(electronic_noise, 0.05, size=self.n_pixels))
        self.update('gain_smearing', self.pixel_id, np.random.normal(gain_smearing, 0.05, size=self.n_pixels))
        self.update('crosstalk', self.pixel_id, np.random.normal(crosstalk, 0.01, size=self.n_pixels))
        self.update('baseline', self.pixel_id, np.random.normal(baseline, 4, size=self.n_pixels))
        self.update('mean_temperature', self.pixel_id, np.random.normal(mean_temperature, 1, size=self.n_pixels))
        self.update('dark_count_rate', self.pixel_id, np.random.normal(dark_count_rate, 0.5, size=self.n_pixels))

    def initialize_simple_camera(self, gain=5.6, electronic_noise=0.8, gain_smearing=0.8, crosstalk=0.07, baseline=500, mean_temperature=19, dark_count_rate=3.):

        self.update('gain', self.pixel_id, np.ones(self.n_pixels)*gain)
        self.update('electronic_noise', self.pixel_id, np.ones(self.n_pixels)*electronic_noise)
        self.update('gain_smearing', self.pixel_id, np.ones(self.n_pixels)*gain_smearing)
        self.update('crosstalk', self.pixel_id, np.ones(self.n_pixels)*crosstalk)
        self.update('baseline', self.pixel_id, np.ones(self.n_pixels)*baseline)
        self.update('mean_temperature', self.pixel_id, np.ones(self.n_pixels)*mean_temperature)
        self.update('dark_count_rate', self.pixel_id, np.ones(self.n_pixels)*dark_count_rate)

if __name__ == '__main__':

    standard_camera = Calibration_Container()
    simple_camera = Calibration_Container()

    standard_camera.initialize_standard_camera()
    simple_camera.initialize_simple_camera()

    standard_camera.save(filename='standard_camera.pk')
    simple_camera.save(filename='simple_camera.pk')
