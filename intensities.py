# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from glob import glob

from h5py import File
from numpy import average, fromiter, arange
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
def process(filename):
    with File(filename, 'r') as f:
        offset = 4000
        bunches = f['/bunches'][...]
        raw = f['/digitizer/channel3'][...].astype('int64')
        norm = average(raw[:, 0:offset], 1)
        tofs = (norm[:, None] - raw)[:, 5000:10000].sum(1)
        ioms = (
            f['/photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc'][...]
                .astype('float64')
        )
        hor_spectra = (
            f['/photon_diagnostics/Spectrometer/hor_spectrum'][...]
                .astype('float64')
        )
        _, n = hor_spectra.shape
        hors = fromiter((average(arange(n), weights=h) for h in hor_spectra),
                        'float')
        ver_spectra = (
            f['/photon_diagnostics/Spectrometer/vert_spectrum'][...]
                .astype('float64')
        )
        _, m = ver_spectra.shape
        vers = fromiter((average(arange(m), weights=v) for v in ver_spectra),
                        'float')
        yield from ({
            'bunch': bunch,
            'tof': tof,
            'iom': iom,
            'hor': hor,
            'ver': ver
        } for bunch, tof, iom, hor, ver
          in zip(bunches, tofs, ioms, hors, vers))


# %%
globbed = glob('/home/ldm/ExperimentalData/Online4LDM/20144078'
               '/Test/Run_224/rawdata/*.h5')
with ProgressBar():
    df = (
        from_sequence(globbed)
            .map(process)
            .flatten()
            .to_dataframe()
            .compute(get=multiprocessing_get)
            .set_index('bunch')
    )

# %%
plt.figure()
plt.subplot(121)
plt.plot(df['iom'], df['tof'], ',')
plt.xlabel('IOM (pC)')
plt.ylabel('TOF (arb units)')
plt.subplot(122)
plt.hist(df['iom'], bins=100)
plt.xlabel('IOM (pC)')
plt.ylabel('Yield (counts)')
plt.tight_layout()
plt.show()

# %%
plt.figure()
where = (df.index % 7 != 0) & (df['iom'] != 0)
plt.hist2d(df['iom'][where], df['tof'][where], bins=(100, 100))
# plt.hist2d(df['iom'][where], df['tof'][where]/df['iom'][where],
#            bins=(linspace(8, 12, 101), linspace(0, 2000, 101)))
# plt.hist2d(df['hor'][where], df['tof'][where], bins=(100, 100))
# plt.hist2d(df['hor'][where], df['tof'][where]/df['iom'][where],
#            bins=(100, linspace(0, 2000, 101)))
# plt.hist2d(df['ver'][where], df['tof'][where], bins=(100, 100))
# plt.hist2d(df['ver'][where], df['tof'][where]/df['iom'][where],
#            bins=(100, linspace(0, 2000, 101)))
plt.grid()
plt.show()
