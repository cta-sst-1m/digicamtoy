from digicamtoy.generator.trace import NTraceGenerator
import numpy as np
import astropy.units as u
from scipy.stats import multivariate_normal

class EllipseGenerator(NTraceGenerator):

    def __init__(self, x_cm, y_cm, width, length, psi, velocity, time_cm, size, geometry, **kwargs):

        self.x_cm = x_cm
        self.y_cm = y_cm
        self.width = width
        self.length = length
        self.psi = psi
        self.velocity = velocity
        self.size = size
        self.pix_x = geometry.pix_x.to(u.mm).value
        self.pix_y = geometry.pix_y.to(u.mm).value
        self.pix_area = geometry.pix_area
        self.time_cm = time_cm
        kwargs.pop('n_pixels', len(self.pix_x))
        # kwargs.pop('n_photons', np.zeros(len(self.pix_x)))
        super(EllipseGenerator, self).__init__(**kwargs)
        self.set_photo_electrons()

    def set_photo_electrons(self):

        aligned_covariance = np.array([
            [self.length ** 2, 0],
            [0, self.width ** 2]
        ])

        rotation = np.array([[np.cos(self.psi), -np.sin(self.psi)],
                     [np.sin(self.psi), np.cos(self.psi)]])
        rotated_covariance = rotation @ aligned_covariance @ rotation.T

        n_photon = multivariate_normal(
            mean=[self.x_cm, self.y_cm],
            cov=rotated_covariance,
        ).pdf(np.column_stack([self.pix_x, self.pix_y]))

        long_x = np.cos(self.psi)
        long_y = np.sin(self.psi)

        longi = (self.x_cm - self.pix_x) * long_x + (self.y_cm - self.pix_y) * long_y

        p = [self.velocity, self.time_cm]
        time_signal = np.polyval(p, longi)
        mask = (time_signal > self.time_end) * (time_signal < self.time_start)

        n_photon[mask] = 0
        n_photon = n_photon * self.size * self.pix_area
        n_photon = np.expand_dims(n_photon, axis=-1)
        time_signal = np.expand_dims(time_signal, axis=-1)

        self.n_photon = n_photon
        self.time_signal = time_signal