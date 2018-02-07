import numpy as np
import time
import ctypes
import os
from numpy.ctypeslib import ndpointer


def histo_naive_numpy(data, bins):

    hist = np.zeros(data.shape[:-1] + (bins.shape[0] - 1,), dtype=int)
    for i in range(hist.shape[0]):

        hist[i] = np.histogram(data[i], bins=bins)[0].astype(int)

    return hist


def histo_C(data, bins):

    histo = np.zeros(data.shape[:-1] + (bins.shape[0] - 1,), dtype=np.uint32)

    lib = np.ctypeslib.load_library("bincount", os.path.dirname(__file__))
    c_bincount = lib.histogram
    c_bincount.restype = None
    c_bincount.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
                           ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                              ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]

    c_bincount(data, histo, bins, data.shape[0], data.shape[1], bins.shape[0])

    return histo


def histo_numpy_C(data, bins):


    hist = np.zeros(data.shape[:-1] + (bins.shape[0],), dtype=np.uint32)
    temp = np.digitize(data, bins=bins).astype(np.uint32)

    lib = np.ctypeslib.load_library("bincount", os.path.dirname(__file__))
    c_bincount = lib.bincount
    c_bincount.restype = None
    c_bincount.argtypes = [ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
                              ndpointer(ctypes.c_uint, flags="C_CONTIGUOUS"),
                              ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]

    c_bincount(temp, hist, temp.shape[0], temp.shape[1], bins.shape[0])
    hist = hist[..., 1:]

    return hist


def hist_numpy(data, bins):
    temp = np.sort(data).ravel()

    hist = np.zeros(data.shape[:-1] + (bins.shape[0],), dtype=np.uint32)
    hist_temp = hist.ravel()

    n_left = np.searchsorted(temp, bins[:-1], side='left')
    n_right = np.searchsorted(temp, bins[1:], side='right')

    hist_temp[n_left:n_right] = 1
    hist.reshape(data.shape)

    return hist


if __name__ == '__main__':

    import bincount_cython

    n_pixels = np.array([2000])
    n_samples = np.logspace(np.log10(10),  np.log10(100), num=5).astype(int)
    n_bins = np.logspace(np.log10(2), 2, num=5).astype(int)
    n_trials = 4
    times = np.zeros((4, ) + n_pixels.shape + n_samples.shape + n_bins.shape)


    print(times.shape)


    """
    for i in range(n_pixels.shape[0]):
        for j in range(n_samples.shape[0]):
            for k in range(n_bins.shape[0]):

                for l in range(n_trials):

                    data = np.random.uniform(0, 4095, size=(n_pixels[i], n_samples[j])).astype(np.float32)
                    # bins = np.linspace(0, 100, 5)
                    bins = np.linspace(-4905, 4095, num=n_bins[k]).astype(np.float32)
                    hist = np.zeros(data.shape[:-1] + (bins.shape[0] - 1,), dtype=int)

                    print('input size : {}, bins size : {}'.format(str(data.shape), str(bins.shape)))

                    t_0 = time.time()
                    old = histo_naive_numpy(data, bins)
                    times[0, i, j, k] += time.time() - t_0
                    # print('time for loop np.histogram : ', times[0, i, j, k])

                    t_0 = time.time()
                    new = histo_numpy_C(data, bins)
                    times[1, i, j, k] += time.time() - t_0
                    # print('time np.digitize + C : ', times[1, i, j, k])

                    t_0 = time.time()
                    c_hist = histo_C(data, bins)
                    times[2, i, j, k] += time.time() - t_0
                    # print('time full C : ', times[2, i, j, k])

                    t_0 = time.time()
                    np.histogram(data, bins=bins)
                    times[3, i, j, k] += time.time() - t_0
                    # print('time flatten data np.histogram : ', times[3, i, j, k])

                    # t_0 = time.time()
                    # c_hist = bincount_cython.histo_cython(data, bins, hist)
                    # times[4, i, j, k] += time.time() - t_0
                    # print('Cython + numpy : ', times[4, i, j, k])

                    # print('Equal histograms ? :', (old == new).all() and (new == c_hist).all())

    times = times / n_trials
    np.savez('time_histo_test.npz', times=times)
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm


    times = np.load('time_histo_test.npz')['times']

    from mpl_toolkits.mplot3d import Axes3D
    plt.figure()
    plt.matshow(np.log10(times[0, 0, :, :]))#, norm=LogNorm(vmin=np.min(times), vmax=np.max(times)))
    plt.xlabel('$N_{smaples}$')
    plt.ylabel('$N_{bins}$')
    plt.colorbar()
    plt.figure()
    plt.matshow(np.log10(times[1, 0, :, :]))#, norm=LogNorm(vmin=np.min(times), vmax=np.max(times)))
    plt.xlabel('$N_{smaples}$')
    plt.ylabel('$N_{bins}$')
    plt.colorbar()
    plt.figure()
    plt.matshow(np.log10(times[2, 0, :, :]))#, norm=LogNorm(vmin=np.min(times), vmax=np.max(times)))
    plt.xlabel('$N_{smaples}$')
    plt.ylabel('$N_{bins}$')
    plt.colorbar()
    plt.figure()
    plt.matshow(np.log10(times[3, 0, :, :]))#, norm=LogNorm(vmin=np.min(times), vmax=np.max(times)))
    plt.xlabel('$N_{smaples}$')
    plt.ylabel('$N_{bins}$')
    plt.colorbar()
    print(times)
    # plt.imshow(times[1, 0, :, :])
   # plt.imshow(times[2, 0, :, :])
    # plt.imshow(times[3, 0, :, :])
    # surf = ax.plot_surface(x, y, times[1, 0, :, :], linewidth=0, antialiased=False)
    # surf = ax.plot_surface(x, y, times[2, 0, :, :], linewidth=0, antialiased=False)
    # surf = ax.plot_surface(x, y, times[3, 0, :, :], linewidth=0, antialiased=False)

    plt.figure()
    plt.loglog(n_pixels, times[0, :, 0, 0], label='1')
    plt.loglog(n_pixels[1:], times[1, :, 0, 0][1:], label='2')
    plt.loglog(n_pixels, times[2, :, 0, 0], label='3')
    plt.loglog(n_pixels, times[3, :, 0, 0], label='4')
    plt.xlabel('$N_{pixels}$')
    plt.ylabel('$t$ [s]')
    plt.legend()

    plt.figure()
    plt.loglog(n_samples, times[0, 0, :, 0], label='1')
    plt.loglog(n_samples[1:], times[1, 0, :, 0][1:], label='2')
    plt.loglog(n_samples, times[2, 0, :, 0], label='3')
    plt.loglog(n_samples, times[3, 0, :, 0], label='4')
    plt.xlabel('$N_{samples}$')
    plt.ylabel('$t$ [s]')
    plt.legend()

    plt.figure()
    plt.loglog(n_bins, times[0, 0, 0, :], label='1')
    plt.loglog(n_bins[1:], times[1, 0, 0, :][1:], label='2')
    plt.loglog(n_bins, times[2, 0, 0, :], label='3')
    plt.loglog(n_bins, times[3, 0, 0, :], label='4')
    plt.xlabel('$N_{bins}$')
    plt.ylabel('$t$ [s]')
    plt.legend()
    plt.show()