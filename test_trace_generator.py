import trace_generator
import matplotlib.pyplot as plt
import numpy as np
from utils.pulse_shape import return_interpolant

f = return_interpolant()
t = np.arange(0, 30, 1)

plt.figure()
plt.plot(t, f(t))

def compute_gain_biais(t_sample, verbose=False):



	gain = 5.6
	sampling_time = 4.
	bins = np.arange(0., 30, sampling_time)

	t_max = t[np.argmax(f(t))]
	t_sample = t_sample

	gain_limit = gain*f(t_max)
	gain_measured = gain*f(t_sample)
	relative_error = (gain-gain_measured)/gain
	
	if verbose:

		print('t_max', t_max)
		print('t_sample', t_sample)
		print('Measured gain : ', gain_measured)
		print('True gain : ', gain)
		print('Gain limit : ', gain_limit)
		print('Relative error : ', (gain-gain_measured)/gain)

		plt.figure()
		plt.plot(t, gain*f(t), label='1 p.e.')
		plt.plot(t, 2*gain*f(t), label='2 p.e.')
		plt.vlines(x=t_max, ymin=0, ymax=2*gain*f(t_max), label='maximum', linestyles='-.', colors='k')
		plt.vlines(x=t_sample, ymin=0, ymax=2*gain*f(t_sample),label='sampling time', linestyles='-.', colors='b')
		plt.legend()


	return (f(t_sample))/f(t)


time_sample = 9
error = compute_gain_biais(time_sample, verbose=True)
plt.show()






