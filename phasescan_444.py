# -*- coding: utf-8 -*-
from glob import glob

from h5py import File
from numpy import average
from cytoolz import concat
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
from dask.multiprocessing import get as multiprocessing_get
import matplotlib.pyplot as plt


# %%
offset, fr, to = 4000, 5750, 6500
path = ('/home/antoine/online4ldm_local/20144078'
        '/Test/Run_{:03d}/rawdata/*.h5').format
runs = {452}  # range(420, 426), 428
globbed = sorted(concat(glob(path(r)) for r in runs))


def spectra(filename):
    with File(filename, 'r') as f:
        bp = f['Background_Period'][...]
        bunches = f['bunches'][...]
        where = bunches % bp != 0
        yield from f['digitizer/channel1'][where, 0:to].astype('float64')


with File(globbed[0]) as f:
    bp = f['Background_Period'].value

with ProgressBar():
    tof = from_sequence(globbed).map(spectra).flatten().mean().compute()

# %%
bins = [
    [5825, 5845],
    [5858, 5880],
    [5890, 5913],
    [5930, 5950],
    [5973, 5995],
    [6025, 6050],
    [6090, 6115],
    [6161, 6186],
    [6256, 6293],
    [6390, 6418]
]
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
            float(f['photon_source/FEL01/PhaseShifter2/DeltaPhase'].value), 2
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
n = len(keys)
good = (
    (df.index % bp != 0)
)
groupped = df[good].dropna().groupby('phase')[['intensity', *keys]]
counts = groupped.count()
cov = groupped.cov()


plt.figure(figsize=(15, 15))
for ix, kx in enumerate(keys):
    for iy, ky in enumerate(keys):
        if ix >= iy:
            continue
        plt.subplot(n, n, n*n-n*(iy+1)+(ix+1))
        pcov = (cov[kx].loc[:, ky] -
                cov[kx].loc[:, 'intensity'] *
                cov[ky].loc[:, 'intensity'] /
                cov['intensity'].loc[:, 'intensity'])
        plt.plot(pcov, '.-')
        plt.ticklabel_format(style='sci', scilimits=[1,1],
                             useOffset=False, axis='y')
        # plt.gca().set_yticks([0])
        plt.twinx()
        plt.plot(counts[kx], 'k.-', alpha=0.2)
        plt.gca().set_yticks([])
        plt.ylim(0, None)
        plt.title('{0}, {1}'.format(ix, iy))
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.plot(tof)
for i, (b0, b1) in enumerate(bins):
    plt.axvspan(b0, b1, 0, 1000, alpha=0.5)
    plt.text((b0+b1) / 2, tof[b0:b1].min(), 'p{}'.format(i),
             horizontalalignment='center')
plt.title('runs={}'.format(runs))
plt.xlim(fr, to)
plt.grid()
plt.show()
