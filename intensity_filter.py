from glob import iglob

from numpy import average, arange, ndarray
from h5py import File
from dask import compute
from dask.multiprocessing import get as multiprocessing_get
from dask.threaded import get as threaded_get
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from cytoolz import memoize
from numba import jit
from scipy.signal import savgol_filter


# %%
offset, fr, to = 4000, 5200, 12000


def spectra(filename):
    try:
        with File(filename, 'r') as f:
#            if not f['user_laser/shutter_opened'].value:
#                yield from ()
#                return
            bp = f['Background_Period'].value
            mods = f['bunches'][...] % bp
            tofs = f['digitizer/channel3'][:, 0:to].astype('float64')
            shifted = (average(tofs[:, 0:offset], axis=1)[:, None] -
                       tofs[:, fr:])
            shifted[shifted < 2] = 0  # threshold
            ioms = f['photon_diagnostics/PM2A/photocurrent'][...]
            for mod, tof, iom in zip(mods, shifted, ioms):
                yield {'mod': mod, 'tof': tof, 'iom': iom}
            return
    except Exception as e:
        print('Error at {}:'.format(filename))
        print(e)
        yield from ()
        return


@memoize
def __average_tof(run, fr=None, to=None):
    if fr is None:
        fr = float('-inf')
    if to is None:
        to = float('inf')
    globbed = iglob(
        '/home/ldm/.gvfs/store on online4ldm.esce/20149046'
        '/Test/Run_{:03d}/rawdata/*.h5'.format(run)
    )
    df = from_sequence(globbed).map(spectra).flatten().to_dataframe({
        'iom': 'float64',
        'mod': 'int64',
        'tof': 'object'
    })
    where = (fr < df['iom']) & (df['iom'] < to)
    with ProgressBar():
        _sg = df[(df['mod'] != 0) & where]['tof'].to_bag()
        _bg = df[(df['mod'] == 0) & where]['tof'].to_bag()
        sg, n, bg, m = compute(
            _sg.mean(),
            _sg.count(),
            _bg.mean(),
            _bg.count(),
            get=threaded_get
        )
        return {'sg': sg, 'sg_n': n, 'bg': bg, 'bg_n': m, 'df': sg - bg}


def average_tof(run, *other_runs, fr=None, to=None):
    loaded = tuple(__average_tof(r, fr=fr, to=to) for r in (run, *other_runs))
    _sg = sum(a['sg'] * a['sg_n'] for a in loaded)
    n = sum(a['sg_n'] for a in loaded)
    sg = _sg / n
    _bg = sum(a['bg'] for a in loaded)
    m = sum(a['bg_n'] for a in loaded)
    bg = _bg / m
    return {'sg': sg, 'sg_n': n, 'bg': bg, 'bg_n': m, 'df': sg - bg}


# %%
@jit
def rebin(n: int, arr: ndarray, axis: int=0):
    m = arr.shape[axis] // n
    sliced = arr.take(range(0, n * m), axis)
    _shape = list(arr.shape)
    _shape.pop(axis)
    _shape.insert(axis, n)
    _shape.insert(axis, m)
    return sliced.reshape(_shape).sum(axis + 1)


# %%
t = arange(fr, to)


@jit
def calib(x, y):
    t0 = 5058
    a = 12050747.724288002
    # upper and lower limits of a
    # 11365539.712968001--12050747.724288002--13116162.619938001
    x = a / (x - t0) ** 2
    y = y / 2 / a * (t - t0) ** 3
    return rebin(5, x) / 5, savgol_filter(rebin(5, y), 5 ** 2, 5)
#    return rebin(10, x) / 10, rebin(10, y)
    # return x, y


# %%
runs = {
#    (365,): '365 off',
#    (366,): '366 off',
#    (367,): '367 off',
    (365, 366, 367): '365--367 off',
    (368,): '368 -1000 fs',
    (369,): '369 0 fs',
#    (370,): '370 1000 fs',
#    (371,): '371 1000 fs',
    (370, 371): '370--371 1000 fs',
}
# tofs = [average_tof(*r, fr=1500, to=2000) for r in runs.keys()]
# tofs = [average_tof(*r, fr=1800, to=1900) for r in runs.keys()]
tofs = [average_tof(*r, fr=1000) for r in runs.keys()]

plt.figure()
for label, tof in zip(runs.values(), tofs):
    plt.plot(*calib(t, tof['df']),
             label='{} n={}+{}'
             .format(label, tof['sg_n'], tof['bg_n']))
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.show()
