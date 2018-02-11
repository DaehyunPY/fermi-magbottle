from itertools import chain
from glob import iglob

from numpy import average, arange, ndarray
from h5py import File
from dask import compute
from dask.bag import from_sequence
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from cytoolz import memoize
from numba import jit


# %%
offset, fr, to = 4000, 5200, 12000


def spectra(filename, bg=False):
    try:
        with File(filename, 'r') as f:
#            if not f['user_laser/shutter_opened'].value:
#                yield from ()
#                return
            bp = f['Background_Period'].value
            bunches = f['bunches'][...]
            where = bunches % bp == 0
            tofs = (f['digitizer/channel3'][where == bg, 0:to]
                    .astype('float64'))
            _shifted = (average(tofs[:, 0:offset], axis=1)[:, None] -
                        tofs[:, fr:])
            _shifted[_shifted < 2] = 0  # threshold
            yield from _shifted
            return
    except Exception as e:
        print('Error at {}:'.format(filename))
        print(e)
        yield from ()
        return


@memoize
def average_tof(run, *other_runs):
    runs = run, *other_runs
    globbed = chain(*(iglob(
        '/home/ldm/.gvfs/store on online4ldm.esce/20149046'
        '/Test/Run_{:03d}/rawdata/*.h5'.format(r)) for r in runs))
    with ProgressBar():
        seq = from_sequence(globbed)
        bg, sg = compute(
            seq.map(spectra, bg=True).flatten().mean(),
            # seq.map(spectra, bg=True).flatten().map(power, 2).mean(),
            seq.map(spectra, bg=False).flatten().mean(),
            # seq.map(spectra, bg=False).flatten().map(power, 2).mean()
        )
        return {
            'bg': bg,  # 'bgerr': (bgsq - bg ** 2) ** 0.5,
            'sg': sg,  # 'sgerr': (sgsq - sg ** 2) ** 0.5,
            'df': sg - bg,  # 'dferr': (bgsq + sgsq - sg ** 2 - bg ** 2) ** 0.5
        }


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
    return rebin(5, x) / 5, rebin(5, y)
    # return x, y


# %%
#labels = {
#    (329, 333, 344,): -1000,  # -13.15
#    (339,): -100,  # -12.25
#    (341,): -70,  # -12.22
#    (324, 325, 326, 327, 328, 346): 0,  # -12.15
#    (340,): 70,  # -12.08
#    (338, 345): 100,  # -12.05
#    (342,): 150,  # -12.00
#    (343,): 200,  # -11.95
#    (331, 332, 334, 335): 500,  # -11.65
#}
#labels = {
#    (350,): '-100 fs',
#    (352,): '-70 fs',
#    (353,): '0 fs',
#    (354,): '30 fs',
#    (351,): '500 fs',
#}
labels = {
#    (358,): '-13.11',
#    (361,): '-12.16',
#    (355, 360): '-12.11',
    (358,): '-13.11',
    (363,): '-10.11',
    (365,): 'off',
    (366,): '366',
}
tofs = {r: average_tof(*r) for r in labels.keys()}

plt.figure()
for label, tof in zip(labels.values(), tofs.values()):
    plt.plot(*calib(t, tof['df']), label=label)
plt.minorticks_on()
plt.grid(which='both')
plt.legend()
plt.show()
