#!/usr/bin/env python
from datetime import datetime
import h5py
import numpy as np
from tqdm import trange

from commandr import command, Run, SetOptions

from digicamtoy.tracegenerator import NTraceGenerator


class ToyEventSink:

    def __init__(self, path, shape, meta=None):
        self.hdf5 = h5py.File(path, 'w')

        self.dset = self.hdf5.create_dataset(
            'adc_count',
            dtype=np.uint16,
            shape=shape,
            chunks=(1, *shape[1:]),
            compression='gzip',
        )

        if meta is not None:
            self.add_meta(meta)
        self.hdf5.attrs['date'] = datetime.utcnow().isoformat()

    def add_meta(self, meta):
        self.hdf5.attrs.update(meta)

    def add_event(self, index, data):
        self.dset[index] = data

    def __del__(self):
        self.hdf5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdf5.close()


@command('main')
def produce_data(
    out_path,
    n_events=100,
    time_start=0,
    time_end=200,
    time_sampling=4,
    n_pixels=1296,
    nsb_rate=0.003,
    crosstalk=0.08,
    gain_nsb=True,
    n_photon=0,
    poisson=True,
    sigma_e=0.8,
    sigma_1=0.8,
    gain=5.8,
    baseline=200,
    time_signal=20,
    jitter=0.3,
    sub_binning=0
):
    n_bins = (time_end - time_start) // time_sampling
    _locals = locals()

    trace_generator = NTraceGenerator(**_locals)
    with ToyEventSink(
        path=out_path,
        shape=(n_events, n_pixels, n_bins),
        meta=_locals
    ) as sink:

        for count in trange(n_events):
            next(trace_generator)
            sink.add_event(index=count, data=trace_generator.adc_count)


if __name__ == '__main__':
    SetOptions(main='main')
    Run()
