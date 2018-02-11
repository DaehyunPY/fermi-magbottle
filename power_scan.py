from glob import glob

from numpy import average
from h5py import File
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from cytoolz import memoize


# %%
offset, fr, to = 4000, 7000, 17000


def spectra(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
        where = bunches % bp != 0
        tofs = f['digitizer/channel1'][where, 0:to].astype('int64')
        yield from average(tofs[:, 0:offset], 1)[:, None] - tofs[:, fr:]


@memoize
def average_tof(run):
    globbed = glob(
        # '/Volumes/store/20144078'
        '/home/ldm/.gvfs/store on online4ldm.esce/20149046'
        '/Test/Run_{:03d}/rawdata/*.h5'.format(run))  # change run number!
    with ProgressBar():
        return from_sequence(globbed).map(spectra).flatten().mean().compute()


# %%
runs = range(86, 88)
# tofs = [average_tof(run)[8500:9500].sum() for run in runs]
tofs = [average_tof(run) for run in runs]
# powers = [5.6, 22, 47, 72, 11, 5.7, 11, 46, 22.2, 69.5, 48, 22, 11, 5.3]

plt.figure()
for run, tof in zip(runs, tofs):
    plt.plot(tof, label='{}'.format(run))
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.show()

# %%
plt.figure()
plt.subplot(121)
plt.plot(runs[0:5], tofs[0:5], label='1st scan')
plt.plot(runs[5:10], tofs[5:10], label='2nd scan')
plt.plot(runs[10:14], tofs[10:14], label='3rd scan')
plt.ylim(0, None)
plt.xlabel('run')
plt.ylabel('yield')

plt.twinx()
plt.plot(runs[0:5], powers[0:5], '--', label='1st scan')
plt.plot(runs[5:10], powers[5:10], '--', label='2nd scan')
plt.plot(runs[10:14], powers[10:14], '--', label='3rd scan')
plt.ylim(0, None)
plt.xlabel('run')
plt.ylabel('power')

plt.subplot(122)
plt.plot(powers[0:5], tofs[0:5], '*', label='1st scan')
plt.plot(powers[5:10], tofs[5:10], '*', label='2nd scan')
plt.plot(powers[10:14], tofs[10:14], '*', label='2nd scan')
plt.xlim(0, None)
plt.ylim(0, None)
plt.xlabel('power')
plt.ylabel('yield')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
