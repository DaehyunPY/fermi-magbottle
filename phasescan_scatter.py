# -*- coding: utf-8 -*-
from glob import glob

from h5py import File
from numpy import average, linspace, ceil
from cytoolz import concat
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
offset, fr, to = 4000, 5700, 7000
path = ('/home/ldm/ExperimentalData/Online4LDM/20144078'
        '/Test/Run_{:03d}/rawdata/*.h5').format
runs = {509}
globbed = sorted(concat(glob(path(r)) for r in runs))

with File(globbed[0]) as f:
    bp = f['Background_Period'].value
    bunches = f['bunches'][...]
    where = bunches % bp != 0
    tof = average(f['digitizer/channel1'][where, 0:to].astype('float64'), 0)
    del bunches, where

# %%
#bins = [
#    [5890, 5913],
#    [6025, 6050],
#    [6256, 6293]
#]

#bins = [
#    [5825, 5845],
#    [5858, 5880],
#    [5890, 5913],
#    [5930, 5950],
#    [5973, 5995],
#    [6025, 6050],
#    [6090, 6115],
#    [6161, 6186],
#    [6256, 6293],
#    [6390, 6418]
#]

bins = [  # 509
    [5844, 5859],
    [5878, 5897],
    [5912, 5930],
    [5954, 5977],
    [6004, 6030],
    [6056, 6098],
    [6129, 6166],
    [6213, 6263],
    [6323, 6391],
    [6494, 6559]
]

#bins = [  # 504
#    [5738, 5747],
#    [5757, 5769],
#    [5778, 5796],
#    [5805, 5818],
#    [5831, 5850],
#    [5862, 5886],
#    [5899, 5918],
#    [5937, 5963],
#    [5980, 6020],
#    [6038, 6063],
#    [6096, 6123]
#]

#bins = [  # 509
#    [5842, 5858],
#    [5875, 5893],
#    [5909, 5942],
#    [5954, 5977],
#    [6000, 6028],
#    [6057, 6094],
#    [6127, 6163],
#    [6215, 6255],
#    [6325, 6392],
#    [6390, 6418],
#    [6494, 6557]
#]
plt.figure()
plt.plot(tof)
for b in bins:
    plt.axvspan(*b, 0, 1000, alpha=0.5)
plt.xlim(fr, to)
plt.minorticks_on()
plt.grid(which='both')
plt.show()


def process(filename):
    with File(filename, 'r') as f:
        bunches = f['bunches'][...]
        phase = round(
            float(f['photon_source/FEL01/PhaseShifter4/DeltaPhase'].value), 2
        )
        intensities = (
            f['photon_diagnostics/FEL01/I0_monitor/iom_sh_a_pc'][...]
                .astype('float64')
        )
        tofs = f['digitizer/channel1'][:, 0:to].astype('float64')
        arrs = average(tofs[:, 0:offset], 1)[:,  None] - tofs
        fmt = 'peak{}'.format
        for bunch, inten, arr in zip(bunches, intensities, arrs):
            yield {
                'bunch': bunch,
                'intensity': inten,
                'phase': phase,
                **{fmt(i): arr[b0:b1].sum() for i, (b0, b1) in enumerate(bins)}
            }


# %%
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
keys = sorted([k for k in df.keys() if k.startswith('peak')])
good = (
    (df.index % bp != 0)
)
phases = df['phase'].unique()
m = int(ceil((len(phases) + 2) ** 0.5))
n = m if m * (m - 1) < len(phases) + 2 else m - 1

# x, y = 3, 4
x, y = 4, 6
kx, ky = (keys[i] for i in (x, y))
xbin, ybin = (linspace(*df[good][k].agg(['min', 'max']), 101)
              for k in (kx, ky))

plt.figure(figsize=(15, 10))
for i, ph in enumerate(phases):
    plt.subplot(n, m, (i + 1))
    where = good & (df['phase'] == ph)
    plt.hist2d(df[where][kx], df[where][ky], bins=(xbin, ybin), cmap='Greys')
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])
    corr = df[where][['intensity', kx, ky]].corr()
    pcorr = ((corr[kx][ky] -
              corr[kx]['intensity'] *
              corr[ky]['intensity']) /
             (1 - corr[kx]['intensity'] ** 2) ** 0.5 /
             (1 - corr[kx]['intensity'] ** 2) ** 0.5)
    plt.title('ph={}, pcorr={:1.2f}'.format(ph, pcorr))
plt.tight_layout()

plt.subplot2grid((n, m), (n - 1, m - 2), colspan=2)
plt.plot(tof)
for i in [x, y]:
    b0, b1 = bins[i]
    plt.axvspan(b0, b1, 0, 1000, alpha=0.5)
    plt.text((b0 + b1) / 2, tof[b0:b1].min(), 'p{}'.format(i),
             horizontalalignment='center')
plt.title('runs={}'.format(runs))
plt.xlim(fr, to)
plt.grid()
plt.show()
