import numpy as np


def gain_drop(nsb_rate, cell_capacitance=85. * 1E-15, bias_resistance = 10. * 1E3):

    return 1. / (1. + nsb_rate * cell_capacitance * bias_resistance * 1E9)


def true_gain_drop(nsb_rate,
                   param=[1.0000, -4.16866e-10, 2.27832e-19, -6.05033e-29]):

    param = np.flip(param, axis=0)

    y = np.polyval(param, nsb_rate)
    y *= (nsb_rate < 1.4 * 1E9)

    return y


def true_xt_drop(nsb_rate, param=[1., -7.41234e-10, 4.60472e-19 , -1.28336e-28]):

    param = np.flip(param, axis=0)

    y = np.polyval(param, nsb_rate)
    y *= (nsb_rate < 1.4 * 1E9)

    return y


def true_pde_drop(nsb_rate, param=[1., -2.2929e-10, 9.59912e-20, -2.22249e-29]):

    param = np.flip(param, axis=0)

    y = np.polyval(param, nsb_rate)
    y *= (nsb_rate < 1.4 * 1E9)

    return y

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    nsb = np.linspace(0, 2*1E9, num=1000)
    g = true_gain_drop(nsb)

    plt.figure()
    plt.plot(nsb, true_gain_drop(nsb), label='Gain drop')
    plt.plot(nsb, true_pde_drop(nsb), label='PDE drop')
    plt.plot(nsb, true_xt_drop(nsb), label='XT drop')
    plt.xlabel('NSB [p.e. Hz]')
    plt.ylabel('[]')
    plt.legend()
    plt.show()

