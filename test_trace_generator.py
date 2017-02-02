import trace_generator
import matplotlib.pyplot as plt
import numpy as np
from utils.pulse_shape import return_interpolant

f = return_interpolant()


x = np.arange(0, 30, 0.001)
plt.figure()
plt.plot(x, f(x))

gain = 3.

t_max = 9.54
t_sample = t_max + 4./np.sqrt(12.)
n_peaks = 10
biais = np.array([(i+1)*gain*f(t_max)-i*gain*f(t_max) for i in range(n_peaks)])/gain

plt.figure()
plt.plot(np.arange(0, len(biais), 1), biais)

plt.figure()
plt.plot(x, 2*gain*f(x)-gain*f(x), label='2 p.e. - 1 p.e.')
plt.plot(x, gain*f(x), label='1 p.e.')
plt.plot(x, 2*gain*f(x), label='2 p.e.')
plt.vlines(x=t_max, ymin=0, ymax=2*gain*f(t_max), label='maximum', linestyles='-.', colors='k')
plt.vlines(x=t_sample, ymin=0, ymax=2*gain*f(t_sample),label='sampling time', linestyles='-.', colors='b')
plt.legend()

print('Measured gain : ', 2*gain*f(t_sample)-gain*f(t_sample))
print('True gain : ', 2*gain*f(t_max)-gain*f(t_max))
print('Relative error : ', (2*gain*f(t_max)-gain*f(t_max) - (2*gain*f(t_sample)-gain*f(t_sample))) / (2*gain*f(t_max)-gain*f(t_max)))

