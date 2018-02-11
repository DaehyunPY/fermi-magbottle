from itertools import chain
from glob import iglob

from numpy import average, arange
from h5py import File
from dask import compute
from dask.multiprocessing import get as multiprocessing_get
from dask.threaded import get as threaded_get
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from cytoolz import memoize
from numba import jit


# %%
offset, fr, to = 4000, 6000, 10000


def spectra(filename):
    try:
        with File(filename, 'r') as f:
            tofs = f['digitizer/channel3'][:, 0:to].astype('int64')
            shifted = average(tofs[:, 0:offset], 1)[:, None] - tofs[:, fr:]
            shtt = f['user_laser/shutter_opened'].value
            for tof in shifted:
                yield {'tof': tof, 'shutter': shtt}
    except Exception as e:
        print('Error at {}:'.format(filename))
        print(e)
        yield from ()


@memoize
def average_tof(run, *other_runs):
    runs = run, *other_runs
    globbed = chain(*(iglob(
        '/home/ldm/.gvfs/store on online4ldm.esce/20149046'
        '/Test/Run_{:03d}/rawdata/*.h5'.format(r)) for r in runs))
    seq = from_sequence(globbed)
    df = seq.map(spectra).flatten().to_dataframe()
    with ProgressBar():
        grp = df.groupby('shutter')['tof']
        n, summed = compute(grp.count(),
                            grp.apply(sum, meta=('tof', 'object')),
                            get=threaded_get)
        return summed / n


# %%
t = arange(fr, to)


@jit
def calib(x, y):
    t0 = 5059
    a = 3.466e3 ** 2
    return a / (x - t0) ** 2, y / 2 / a * (t - t0) ** 3
    # return x, y


# %%
runs = [188, 189, 191, 192, 193, 194, 195]
tofs = [average_tof(run) for run in runs]

plt.figure(figsize=(5, 4))
for run, tof in zip(runs, tofs):
    if run not in {189, 191, 192}:
        plt.plot(*calib(t, tof[False]), label='{} off'.format(run))
    if run not in {188, 195}:
        plt.plot(*calib(t, tof[True]), label='{} on'.format(run))
    # plt.plot(*calib(t, tof[True] - tof[False]), label='{} diff'.format(run))
plt.xlabel('KE (eV)')
#plt.xlabel('TOF (ns)')
plt.ylabel('yield')
#plt.ylim(-100, 200)
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.tight_layout()
plt.show()


# %%
runs = [181, 182, 183]
tof = average_tof(*runs)

plt.figure(figsize=(5, 4))
plt.plot(*calib(t, tof[False]), label='off')
plt.plot(*calib(t, tof[True]), label='on')
plt.plot(*calib(t, tof[True] - tof[False]), label='diff')
plt.title(','.join(str(r) for r in runs))
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.tight_layout()
plt.show()
