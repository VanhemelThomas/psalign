import numpy as np
import numba as nb

from .utils import searchsorted_merge

# Code adapted from https://stackoverflow.com/questions/67466241/full-algorithm-math-of-natural-cubic-splines-computation-in-python/67466242#67466242

# Solves linear system given by Tridiagonal Matrix
# Helper for calculating cubic splines
nb.core.entrypoints.init_all = lambda: None
fastmath = True

@nb.njit([f'f{ii}[:](f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:])' for ii in (8,)], cache = True, fastmath = fastmath, inline = 'always')
def tri_diag_solve(A, B, C, F):
    n = B.size
    # assert A.ndim == B.ndim == C.ndim == F.ndim == 1 and (A.size == B.size == C.size == F.size == n) #, (A.shape, B.shape, C.shape, F.shape)
    Bs, Fs = np.zeros_like(B), np.zeros_like(F)
    Bs[0], Fs[0] = B[0], F[0]
    for i in range(1, n):
        Bs[i] = B[i] - A[i] / Bs[i - 1] * C[i - 1]
        Fs[i] = F[i] - A[i] / Bs[i - 1] * Fs[i - 1]
    x = np.zeros_like(B)
    x[-1] = Fs[-1] / Bs[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (Fs[i] - C[i] * x[i + 1]) / Bs[i]
    return x
    
# Calculate cubic spline params
@nb.njit([f'Tuple((f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:]))(f{ii}[:], f{ii}[:])' for ii in (8,)], cache = True, fastmath = fastmath, inline = 'always')
def calc_spline_params(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = y
    h = np.diff(np.ascontiguousarray(x))
    c = np.concatenate((np.zeros((1,), dtype = y.dtype),
        np.append(tri_diag_solve(h[:-1], (h[:-1] + h[1:]) * 2, h[1:],
        ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1]) * 3), y.dtype.type(0))))
    d = np.diff(c) / (3 * h)
    b = (a[1:] - a[:-1]) / h + (2 * c[1:] + c[:-1]) / 3 * h
    return a[1:], b, c[1:], d
    
# Spline value calculating function, given params and "x"
@nb.njit([f'f{ii}[:](f{ii}[:], i4[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:])' for ii in (8,)], cache = True, fastmath = fastmath, inline = 'always')
def func_spline(x: np.ndarray, ix: np.ndarray, x0: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    dx = x - x0[1:][ix]
    return a[ix] + (b[ix] + (c[ix] + d[ix] * dx) * dx) * dx
    
# Compute piece-wise spline function for "x" out of sorted "x0" points
@nb.njit([f'f{ii}[:](f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:])' for ii in (8,)], cache = True, fastmath = fastmath, inline = 'always')
def piece_wise_spline(x: np.ndarray, x0: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    xsh = x.shape
    # x = x.ravel()
    ix = searchsorted_merge(x0[1 : -1], x)
    y = func_spline(x, ix, x0, a, b, c, d)
    y = y.reshape(xsh)
    return y

@nb.njit([f'f{ii}[:](f{ii}[:], f{ii}[:], f{ii}[:])' for ii in (4, 8)], cache = True, fastmath = fastmath, inline = 'always')
def cubic_spline(x0: np.ndarray, y0: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, c, d = calc_spline_params(x0, y0)
    return piece_wise_spline(x, x0, a, b, c, d)


