# -*- coding: utf-8 -*-
from glob import glob

from h5py import File
from numpy import average, histogram, arange
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


# %%
def spectra(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
        where = bunches%bp != 0
        yield from f['digitizer/channel3'][where, ...].astype('int64')


#globbed = glob('/Volumes/store/20144078'
#               '/Test/Run_132/rawdata/*.h5')  # change run number!
globbed = glob('/Users/daehyun/Desktop/*.h5')  # change run number!
with ProgressBar():
    tof = from_sequence(globbed).map(spectra).flatten().mean().compute()

# %%
plt.figure()
plt.plot(tof)
plt.grid()
plt.show()


# %%
def process(filename):
    with File(filename, 'r') as f:
        offset = 4000
        bins = [5895, 5930, 5975, 6035, 6075]  # double check bins!
        to = bins[-1]
        bp = f['Background_Period'].value
        phase = round(
            float(f['photon_source/FEL01/PhaseShifter5/DeltaPhase'].value), 2
        )
        bunches = f['bunches'][...]
        where = bunches%bp != 0
        tofs = f['digitizer/channel3'][where, 0:to].astype('int64')
        intensities = (
            f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc'][where]
                .astype('float64')
        )
        norm = average(tofs[:, 0:offset], 1)
        arrs = (norm[:, None]-tofs)/intensities[:, None]
        n, m = arrs.shape
        idx = arange(m)
        fmt = 'peak{}'.format
        for bunch, arr in zip(bunches[where], arrs):
            hist, *_ = histogram(idx, bins, weights=arr)
            yield {
                'bunch': bunch,
                'phase': phase,
                **{fmt(i): h for i, h in enumerate(hist)}
            }

# %%
with ProgressBar():
    df = (
        from_sequence(globbed)
            .map(process)
            .flatten()
            .to_dataframe()
            .compute()
    )


# %%
plt.figure()
keys = sorted([k for k in df.keys() if k.startswith('peak')])
n = len(keys)
cov = df.groupby('phase')[keys].cov()
for i in range(n):
    for j in range(n):
        if i > j:
            continue
        plt.subplot(n, n, n*n-n*(j+1)+(i+1))
        plt.plot(cov[keys[i]].loc[:, keys[j]])
plt.show()