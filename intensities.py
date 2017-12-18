# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from glob import glob

from h5py import File
from numpy import average
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


# %%
def process(filename):
    with File(filename, 'r') as f:
        offset = 4000
        bunches = f['bunches'][...]
        raw = f['digitizer/channel3'][...].astype('int64')
        norm = average(raw[:, 0:offset], 1)
        tofs = (norm[:, None]-raw)[:, 5000:10000].sum(1)
        ioms = (
            f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc'][...]
                .astype('float64')
        )
        yield from ({
            'bunch': bunch,
            'tof': tof,
            'iom': iom
        } for bunch, tof, iom in zip(bunches, tofs, ioms))


# %%
globbed = glob('/Volumes/store/20144078/Test/Run_127/rawdata/*.h5')
with ProgressBar():
    df = (
        from_sequence(globbed)
            .map(process)
            .flatten()
            .to_dataframe()
            .compute()
            .set_index('bunch')
    )

# %%
plt.figure()
# plt.plot(df.index, df['iom'], ',')
# plt.hist(df['iom'], bins=100)
plt.plot(df['iom'], df['tof'], ',')
plt.show()
