import multiprocessing as mp
import numpy as np
import numba as nb
import time

from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Callable
from tqdm import tqdm

nb.core.entrypoints.init_all = lambda: None

def l2_normalize(data):
    # l2 normalize every row of the data matrix
    for i in range(data.shape[0]):
        data[i, :] /= np.linalg.norm(data[i, :])
    return data

def tic_normalize(data):
    # TIC normalize every row of the data matrix and afterwards compute the average row
    for i in range(data.shape[0]):
        if np.sum(data[i, :]) > 0:
            data[i, :] /= np.sum(data[i, :])
    return data

def tic_normalize_and_average_pixel(data):
    # TIC normalize every row of the data matrix and afterwards compute the average row
    for i in range(data.shape[0]):
        if np.sum(data[i, :]) > 0:
            data[i, :] /= np.sum(data[i, :])
    reference = np.average(data, axis=0)
    return data, reference / np.sum(reference)

def average_pixel(data):
    # Compute average row after TIC normalization
    tic = np.sum(data, axis=1)
    reference = np.average((1 / tic).reshape((-1, 1)) * data, axis=0)
    return reference / np.sum(reference)

def f_star(f_args):
    # Utility function for using mp.starmap with tqdm
    # f_args contains a tuple of (f, args)
    f = f_args[0]
    args = f_args[1:]
    return f(*args)

def f_star_i(f_args):
    # f_args contains a tuple of (f, args)
    f = f_args[0]
    args = f_args[1: -1]
    i = f_args[-1]
    return i, f(*args)

def execute_indexed_parallel(f: Callable, *, args: list, tqdm_args: dict=None, nb_cores: int=None):
    # Perform f in parrallel using mp.Pool, the return values are stored in a list containing (i, f(*args[i])) BUT NOT IN NECESSARILY IN ASCENDING ORDER OF i
    if nb_cores is None:
        nb_cores = mp.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=nb_cores) as executor:
        if tqdm_args is None:
            tqdm_args = {}
        result = [None for _ in range(len(args))]
        with tqdm(total=len(args), **tqdm_args) as pbar:
            futures = {executor.submit(f, *args[i]): i for i in range(len(args))}
            for future in as_completed(futures):
                index = futures[future]
                result[index] = future.result()
                pbar.update(1)
    return result

def execute_indexed_parallel_executor(executor: ProcessPoolExecutor, f: Callable, *, args: list, tqdm_args: dict=None):
    # Perform f in parrallel using mp.Pool, the return values are stored in a list containing (i, f(*args[i])) BUT NOT IN NECESSARILY IN ASCENDING ORDER OF i
    if tqdm_args is None:
        tqdm_args = {}
    result = [None for _ in range(len(args))]
    with tqdm(total=len(args), **tqdm_args) as pbar:
        futures = {executor.submit(f_star_i, (f, *arg, i)): i for i, arg in enumerate(args)}
        for future in as_completed(futures):
            index = futures[future]
            result[index] = future.result()
            pbar.update(1)
    return result

def execute_parallel_along_axis(out: np.ndarray, func: Callable, *, args: list, nb_cores: int=None, precompile: bool=False, tqdm_args: dict=None):
    # Perform func in parallelon nb_cores cores. Out is an array of appropriate shape which will contain the results of f(out[i], *args[i]) if the args has the same number of elements as out.shape[0], otherwise it contains the results f(out[i], args). It can precompile the function func if precompile is True and func is a Numba function.
    if tqdm_args is None:
        tqdm_args = {}
        
    if nb_cores is None:
        nb_cores = mp.cpu_count() // 2

    with ProcessPoolExecutor(max_workers=nb_cores) as executor:
        
        if precompile:
            print('Compiling Numba functions: this can take several seconds.', end='')
            s = time.perf_counter_ns()
            out_precompile = np.copy(out[: nb_cores])
            if len(args) != out.shape[0]:
                futures = {executor.submit(func, out_precompile[i], *args): i for i in range(nb_cores)}
            else:
                futures = {executor.submit(func, out_precompile[i], *args[i]): i for i in range(nb_cores)}
            for future in as_completed(futures):
                index = futures[future]
                out_precompile[index] = future.result()
            del futures
            del out_precompile
            print("\r", end="")
            print(f'Compiling Numba functions: finished in {np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2)} seconds!        ')
            
        s = time.perf_counter_ns()
        with tqdm(total=out.shape[0], **tqdm_args) as pbar:
            if len(args) != out.shape[0]:
                futures = {executor.submit(func, out[i], *args): i for i in range(out.shape[0])}
            else:
                futures = {executor.submit(func, out[i], *args[i]): i for i in range(out.shape[0])}
            for future in as_completed(futures):
                index = futures[future]
                out[index] = future.result()
                del futures[future]
                pbar.update(1)

    return s

def execute_parallel_along_axis_lists(out1: list, out2: list, inp: np.ndarray, func: Callable, *, args: list, nb_cores: int=None, precompile: bool=False, tqdm_args: dict=None):
    # Perform func in parallelon nb_cores cores. Out is an array of appropriate shape which will contain the results of f(out[i], *args[i]) if the args has the same number of elements as out.shape[0], otherwise it contains the results f(out[i], args). It can precompile the function func if precompile is True and func is a Numba function.
    if tqdm_args is None:
        tqdm_args = {}
        
    if nb_cores is None:
        nb_cores = mp.cpu_count() // 2

    with ProcessPoolExecutor(max_workers=nb_cores) as executor:
        
        if precompile:
            print('Compiling Numba functions: this can take several seconds.', end='')
            s = time.perf_counter_ns()
            out_precompile = np.copy(inp[: nb_cores])
            if len(args) != inp.shape[0]:
                futures = {executor.submit(func, out_precompile[i], *args): i for i in range(nb_cores)}
            else:
                futures = {executor.submit(func, out_precompile[i], *args[i]): i for i in range(nb_cores)}
            for future in as_completed(futures):
                index = futures[future]
            del futures
            del out_precompile
            print("\r", end="")
            print(f'Compiling Numba functions: finished in {np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2)} seconds!        ')
            
        s = time.perf_counter_ns()
        with tqdm(total=inp.shape[0], **tqdm_args) as pbar:
            if len(args) != inp.shape[0]:
                futures = {executor.submit(func, inp[i], *args): i for i in range(inp.shape[0])}
            else:
                futures = {executor.submit(func, inp[i], *args[i]): i for i in range(inp.shape[0])}
            for future in as_completed(futures):
                index = futures[future]
                x, y = future.result()
                out1[index] = x
                out2[index] = y
                del futures[future]
                pbar.update(1)

    return s

def execute_parallel_along_axis_pool(out: np.ndarray, func: Callable, *, args: list, executor: ProcessPoolExecutor, tqdm_args: dict=None):
    # Perform func in parallelon nb_cores cores. Out is an array of appropriate shape which will contain the results of f(out[i], *args[i]) if the args has the same number of elements as out.shape[0], otherwise it contains the results f(out[i], args). It can precompile the function func if precompile is True and func is a Numba function.
    if tqdm_args is None:
        tqdm_args = {}
            
    s = time.perf_counter_ns()
    with tqdm(total=out.shape[0], **tqdm_args) as pbar:
        if len(args) != out.shape[0]:
            futures = {executor.submit(func, out[i], *args): i for i in range(out.shape[0])}
        else:
            futures = {executor.submit(func, out[i], *args[i]): i for i in range(out.shape[0])}
        for future in as_completed(futures):
            index = futures[future]
            out[index] = future.result()
            del futures[future]
            pbar.update(1)

    return s

def compile_numba(f, *args):
    print('Compiling Numba functions: this can take several seconds.', end='')
    s = time.perf_counter_ns()
    f(*args)
    print("\r", end="")
    print(f'Compiling Numba functions: finished in {np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2)} seconds!        ')

@nb.njit(cache=True)
def searchsorted_merge(a, b):
    # Compute the indices of the elements in b that should be inserted into a to maintain order.
    ix = np.zeros((len(b),), dtype=np.int32)
    pa, pb = 0, 0
    while pb < len(b):
        if pa < len(a) and a[pa] < b[pb]:
            pa += 1
        else:
            ix[pb] = pa
            pb += 1
    return ix

@nb.njit(cache=True)
def searchsorted_merge_right(a, b):
    # Compute the indices of the elements in b that should be inserted into a to maintain order.
    ix = np.zeros((len(b),), dtype=np.int32)
    pa, pb = 0, 0
    while pb < len(b):
        if pa < len(a) and a[pa] <= b[pb]:
            pa += 1
        else:
            ix[pb] = pa
            pb += 1
    return ix
                      
@nb.njit(cache=True)
def binning(x_correct, x_wrong, values):
    # Bin the values in x_wrong to the bins defined by x_correct. x_wrong is assumed to be sorted. The values are divided between the bins according to the distance to the bin edges.
    
    index_min = np.argmin(np.abs(x_wrong - x_correct[0]))
    if x_wrong[index_min] > x_correct[0]:
        index_min -= 1
        index_min = index_min if index_min > 0 else 0
    index_max = np.argmin(np.abs(x_wrong - x_correct[-1]))
    if x_wrong[index_max] < x_correct[-1]:
        index_max += 1
        index_max = index_max if index_max < x_wrong.shape[0] else x_wrong.shape[0]
        
    result = np.zeros_like(x_correct, dtype=values.dtype)
    
    values = values[index_min: index_max + 1]
    x_wrong = x_wrong[index_min: index_max + 1]
    
    indices = searchsorted_merge_right(x_correct, x_wrong) - 1
    
    idxs = indices < 0
    if idxs.sum() > 0:
        result[0] = np.dot(values[idxs], x_wrong[idxs] - x_correct[0]) / (x_correct[0] - x_correct[1])
    
    idxs = np.logical_and(0 <= indices, indices < x_correct.shape[0] - 1)
    if idxs.sum() > 0:
        temp = indices[idxs]
        shifted = temp + 1
        factor = np.divide(x_wrong[idxs] - x_correct[temp], x_correct[shifted] - x_correct[temp])
        vals = values[idxs]
        for i, index in enumerate(temp):
            result[index] += vals[i] * (1 - factor[i])
            result[index + 1] += vals[i] * factor[i]
    
    idxs = indices >= x_correct.shape[0] - 1
    if idxs.sum() > 0:
        temp = indices[idxs]
        result[-1] += np.dot(values[idxs], np.divide(x_wrong[idxs] - x_correct[temp], x_correct[temp] - x_correct[temp - 1]))
                
    return result

@nb.njit
def interpolation(x_correct, x_wrong, values):
    return np.interp(x_correct, x_wrong, values)

def binning_matrix(x_correct, x_wrong, values):
    # Matrix version of binning where every row is transformed identically.
    
    index_min = np.argmin(np.abs(x_wrong - x_correct[0]))
    if x_wrong[index_min] > x_correct[0]:
        index_min -= 1
        index_min = np.max([index_min, 0])
    index_max = np.argmin(np.abs(x_wrong - x_correct[-1]))
    if x_wrong[index_min] < x_correct[-1]:
        index_max += 1
        index_max = np.min([index_max, x_wrong.shape[0]])
        
    result = np.zeros((values.shape[0], x_correct.shape[0]), dtype=values.dtype)
    
    values = values[:, index_min: index_max + 1]
    x_wrong = x_wrong[index_min: index_max + 1]
    
    indices = np.searchsorted(x_correct, x_wrong, side='right') - 1
    
    idxs = indices < 0
    if idxs.sum() > 0:
        result[:, 0] = np.dot(values[:, idxs], x_wrong[idxs] - x_correct[0]) / (x_correct[0] - x_correct[1])
    
    idxs = np.logical_and(0 <= indices, indices < x_correct.shape[0] - 1)
    if idxs.sum() > 0:
        temp = indices[idxs]
        shifted = temp + 1
        factor = np.divide(x_wrong[idxs] - x_correct[temp], x_correct[shifted] - x_correct[temp])
        vals = values[:, idxs]
        for i, index in enumerate(temp):
            result[:, index] += vals[:, i] * (1 - factor[i])
            result[:, index + 1] += vals[:, i] * factor[i]
    
    idxs = indices >= x_correct.shape[0] - 1
    if idxs.sum() > 0:
        temp = indices[idxs]
        result[:, -1] += np.dot(values[:, idxs], np.divide(x_wrong[idxs] - x_correct[temp], x_correct[temp] - x_correct[temp - 1]))
        
    return result
                    
                

                