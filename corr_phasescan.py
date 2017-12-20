# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from glob import glob

from h5py import File
from numpy import average, histogram, arange, array, linspace
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


globbed = glob('/Volumes/store/20144078'
               '/Test/Run_132/rawdata/*.h5')  # change run number!
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
        bp = f['Background_Period'][...]
        phase = f['photon_source/FEL01/PhaseShifter5/DeltaPhase'][...]
        bunches = f['bunches'][...]
        where = bunches%bp != 0
        tofs = f['digitizer/channel3'][where, 0:to].astype('int64')
        intensities = (f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc']
                        [where].astype('float64'))
        norm = average(tofs[:, 0:offset], 1)
        arrs = (norm[:, None]-tofs)/intensities[:, None]
        n, m = arrs.shape
        for bunch, arr in zip(bunches[where], arrs):
            hist = histogram(arange(m), bins, weights=arr)[0]
            yield {
                'bunch': bunch,
                'phase': phase,
                {}
            }

# %%
    from_sequence(globbed).map(process).flatten().take(10)

# %%
with ProgressBar():
    events = array(
        from_sequence(globbed)
            .map(process)
            .flatten()
            .compute()
    )


# %%
def bins(idx):  # rescale region of interest!
#    return 100
    if idx == 0:
        return linspace(0, 50, 101)
    elif idx == 1:
        return linspace(50, 250, 101)
    elif idx == 2:
        return linspace(500, 1000, 101)
    elif idx == 3:
        return linspace(50, 150, 101)
    else:
        return 100


plt.figure()
_, n, *_ = events.shape
for i in range(n):
    for j in range(n):
        # if i > j:
        #     continue
        plt.subplot(n, n, n*n-n*(j+1)+(i+1))
        xbins = bins(i)
        ybins = bins(j)
        plt.hist2d(events[:, i], events[:, j], bins=(xbins, ybins),
                   cmap='Greys')
        plt.axis('equal')
        # plt.title('{} {}'.format(i, j))
plt.show()
