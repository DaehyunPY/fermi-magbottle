from glob import glob

from numpy import average
from h5py import File
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from cytoolz import memoize


# %%
offset, fr, to = 4000, 5500, 6500


def spectra(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
        where = bunches % bp != 0
        tofs = f['digitizer/channel3'][where, 0:to].astype('int64')
        yield from average(tofs[:, 0:offset], 1)[:, None] - tofs[:, fr:]


@memoize
def average_tof(run):
    globbed = glob(
        # '/Volumes/store/20144078'
        '/home/ldm/ExperimentalData/Online4LDM/20144078'
        '/Test/Run_{:03d}/rawdata/*.h5'.format(run))  # change run number!
    with ProgressBar():
        return from_sequence(globbed).map(spectra).flatten().mean().compute()


# %%
runs = {224, 297}
tofs = [average_tof(run) for run in runs]

plt.figure()
for run, tof in zip(runs, tofs):
    plt.plot(tof, label='{}'.format(run))
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.show()
