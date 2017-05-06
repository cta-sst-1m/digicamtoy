import numpy as np
import pickle
import datetime

class Calibration_Container():

    """
    a container with all calibration parameters for SST-1M camera
    each field contains a dict() of np.array
    """

    def __init__(self, n_pixels=1296, filename=None):

        if filename is not None:

            self._load(filename)

        else:

            self.n_pixels = n_pixels
            self.pixel_id = [i for i in range(n_pixels)]

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


    camera_container = Calibration_Container()

    camera_container.initialize_standard_camera()
    camera_container.save(filename='standard_camera.pk')
    camera_container.initialize_simple_camera(dark_count_rate=0.)
    camera_container.save(filename='simple_camera.pk')

    cluster_7_container = Calibration_Container(n_pixels=3*7)
    cluster_7_container.initialize_standard_camera(dark_count_rate=0.)
    cluster_7_container.save(filename='standard_cluster_7.pk')
    cluster_7_container.initialize_simple_camera(dark_count_rate=0.)
    cluster_7_container.save(filename='simple_cluster_7.pk')

    cluster_19_container = Calibration_Container(n_pixels=9*3)
    cluster_19_container.initialize_standard_camera(dark_count_rate=0.)
    cluster_19_container.save(filename='standard_cluster_19.pk')
    cluster_19_container.initialize_simple_camera(dark_count_rate=0.)
    cluster_19_container.save(filename='simple_cluster_19.pk')






