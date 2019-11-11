from digicamtoy.generator.trace import NTraceGenerator
import numpy as np
import astropy.units as u


class EllipseGenerator(NTraceGenerator):

    def __init__(self, x_cm, y_cm, width, length, psi, velocity, time_cm, size, geometry, **kwargs):

        self.x_cm = x_cm
        self.y_cm = y_cm
        self.width = width
        self.length = length
        self.psi = psi
        self.velocity = velocity
        self.size = size
        self.geometry = geometry
        self.time_cm = time_cm
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
        ).pdf(np.column_stack([self.geometry.pix_x, self.geometry.pix_y]))

        longi, trans = self.geometry.get_shower_coordinates(
            self.x_cm,
            self.y_cm,
            self.psi
        )

        p = [self.velocity, self.time_cm]
        time_signal = np.polyval(p, longi)

        self.n_photon = n_photon * self.size * self.geometry.pix_area
        self.time_signal = time_signal
