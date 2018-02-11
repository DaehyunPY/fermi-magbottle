# -*- coding: utf-8 -*-
from glob import iglob
from itertools import chain

from numpy import arange, average
from h5py import File
from dask import compute
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from cytoolz import memoize


# %%
offset, fr, to = 4000, 6000, 15000


def spectra(filename, bg=False):
    with File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
        where = bunches % bp == 0
        tofs = f['digitizer/channel1'][where == bg, 0:to].astype('int64')
        yield from average(tofs[:, 0:offset], axis=1)[:, None] - tofs[:, fr:]


@memoize
def average_tof(run, *other_runs):
    runs = run, *other_runs
    globbed = chain(*(iglob(
        '/home/ldm/.gvfs/store on online4ldm.esce/20149046'
        '/Test/Run_{:03d}/rawdata/*.h5'.format(r)) for r in runs))
    with ProgressBar():
        seq = from_sequence(globbed)
        bg, sg = compute(seq.map(spectra, bg=True).flatten().mean(),
                         seq.map(spectra, bg=False).flatten().mean())
        return {'bg': bg, 'sg': sg, 'df': sg - bg}


# %%
t = arange(fr, to)
# t0 = 5059
# a = 3.46629e-3 ** 0.5
# m = a * (t - t0) ** 2

# runs = range(172, 176)
# runs = range(176, 181)
runs = [324, 325]
tofs = [average_tof(run) for run in runs]

plt.figure()
for run, tof in zip(runs, tofs):
    plt.plot(t, tof['df'], label=run)
    # plt.plot(m, tof['df'] / 2 / a / (t - t0), label=run)
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.show()
