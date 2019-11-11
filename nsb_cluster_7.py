from digicamtoy.generator.trace import NTraceGenerator
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    nsb = np.linspace(1, 1.5, 3)
    n_events = 10
    cluster_lsb = np.zeros((len(nsb), n_events, 50))
    trigger_prob = [None] * len(nsb)
    threshold = [None] * len(nsb)

    n_pixels = 21
    baseline = np.zeros((len(nsb), n_pixels))

    for j, rate in enumerate(nsb):
        toy = NTraceGenerator(n_pixels=n_pixels, nsb_rate=rate)

        for k in range(1000):

            baseline[j] += np.mean(toy.adc_count, axis=-1)
        baseline = baseline / 1000

        # plt.figure()
        for i in range(n_events):
            toy.next()
            trace = toy.adc_count
            trace = trace - baseline[j]
            trace[trace < 0] = 0
        #        plt.plot(np.arange(0, trace.shape[-1]), trace.T)
            trigger_trace = np.sum(trace, axis=0)
        #        plt.plot(trigger_trace)
            cluster_lsb[j, i] = trigger_trace

        temp = cluster_lsb.reshape(cluster_lsb.shape[0],
                                   cluster_lsb.shape[1] * cluster_lsb.shape[2])
        hist = np.histogram(temp, bins='auto')
        bins = hist[1]
        hist = hist[0]
        bins = bins[:-1]
        hist = hist / np.sum(hist)
        trigger_prob[j] = 1 - np.cumsum(hist)
        threshold[j] = bins / 21 / toy.gain[0]
        print(toy.gain)

    plt.figure()
    for i in range(len(threshold)):
        plt.semilogy(threshold[i],  trigger_prob[i])
    plt.show()
