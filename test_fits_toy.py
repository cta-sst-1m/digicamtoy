import h5py
#from .DigiCamCommissioning.utils.histogram import Histogram
import numpy as np
from DigicamCommissionning import utils.Histogram

if __name__ == '__main__':

    directory = './data_calibration_cts/'

    n_traces = 5000
    dac_list =  np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450,460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600])
    mu_list = np.arange(0, 41, 1)
    n_dac = len(dac_list)


    bin_centers = np.arange(0, 4000, 1)

    for mu in mu_list:

        path = directory + 'toy_data_poisson_signal_' + str(mu) + '.hdf5'
        mc_data_file = h5py.File(path, 'r')
        print(mc_data_file['data']['traces'].shape)
        n_traces_tot = mc_data_file['data']['traces'].shape[0]
        n_samples = mc_data_file['data']['trace'].shape[1]
        adc_value = np.zeros(n_traces)

        for i, index in enumerate(np.random.randint(low=0, high=n_traces_tot, size=n_traces)):

            photon_index = int((n_samples-1.)/2.)
            adc_value[i] = np.max(mc_data_file['data']['traces'][index, photon_index-2:photon_index+2:1])



        #Histogram()