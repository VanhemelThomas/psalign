import numpy as np
import numba as nb
import time

from bisect import bisect_left, bisect_right
from tqdm import tqdm
from scipy.optimize import fmin_l_bfgs_b
from concurrent.futures import ProcessPoolExecutor

from .utils import average_pixel, binning, execute_parallel_along_axis, searchsorted_merge, compile_numba, binning_matrix, execute_parallel_along_axis_pool, execute_parallel_along_axis_lists
from .mass_dispersion import compute_mass_dispersion
from .cubic_spline import cubic_spline
from .inliers import compute_inliers

nb.core.entrypoints.init_all = lambda: None
fastmath = True

@nb.njit(cache=True, inline='always', fastmath=fastmath)
def _template_matching(template: np.ndarray, signal: np.ndarray):
    # Perform template mathcingg using the dot product between the template and the signal.
    result = np.empty((signal.shape[0] - template.shape[0],), dtype=signal.dtype)
    for i in range(result.shape[0]):
        result[i] = np.dot(template * np.max(signal[i: i + template.shape[0]]), signal[i: i + template.shape[0]])
    return np.argmax(result)

@nb.njit(cache=True, inline='always', fastmath=fastmath)
def _get_fraction_votes(k: int, votes: np.ndarray):
    # Get the most possible votes if a sliding window of size k is used and return the fraction of inlier votes / (most possible votes) for each entry of the array.
    most_possible_votes = k * np.ones_like(votes)
    for i in range(k):
        most_possible_votes[i] -= ((k - 1) - i)
    for i in range(most_possible_votes.shape[0] - k, most_possible_votes.shape[0]):
        most_possible_votes[i] -= (k - (most_possible_votes.shape[0] - i))
    return np.divide(votes, most_possible_votes)

def _robust_inlier_detection(axis: np.ndarray, array: np.ndarray, residual_threshold_min: float, residual_threshold_max: float):
    # Every entry of the array will be part of a sliding window of size k, they will get an inlier/outlier vote per step of the sliding window (if they are part of the sliding window). The fraction of the inlier / (outlier + inlier) votes will be returned per entry of the array.
    k = 5
    votes = np.zeros_like(axis)
    residual_threshold = np.mean(np.abs(array - np.mean(array)))
    residual_threshold = residual_threshold_min if residual_threshold_min > residual_threshold else residual_threshold
    residual_threshold = residual_threshold_max if residual_threshold_max < residual_threshold else residual_threshold
    for i in range(axis.shape[0] - k + 1):
        ax = axis[i: i + k]
        d = array[i: i + k]
        inlier_idxs = compute_inliers(ax.reshape(-1, 1), d, residual_threshold=residual_threshold, random_state=0)
        votes[i + inlier_idxs] += 1
    return _get_fraction_votes(k, votes)

@nb.njit(cache=True, inline='always', fastmath=fastmath)
def _get_matches(reference: np.ndarray, data: np.ndarray, nb_segments: int, window: int, factor: float):
    # Get the initial estimate for the warping knots
    ind = np.empty((nb_segments,), dtype=np.int32)
    template_ind = np.empty(ind.shape, dtype=ind.dtype)
    enlarged_window = int(factor * window)
    
    for j in range(nb_segments):
        start = int(j * data.shape[0] / nb_segments) + enlarged_window
        end = int((j + 1) * data.shape[0] / nb_segments) - enlarged_window
        if j == 0:
            start = enlarged_window
        elif j == nb_segments - 1:
            end = -enlarged_window
        template_ind[j] = np.argmax(data[start: end]) + start
        
        if data[template_ind[j]] <= 1e-5:
            ind[j] = template_ind[j]
        else:
            data_window = data[template_ind[j] - window: template_ind[j] + window + 1]
            ind[j] = _template_matching(data_window / data[template_ind[j]], reference[template_ind[j] - enlarged_window: template_ind[j] + enlarged_window + 1]) + template_ind[j] - enlarged_window
    
    return ind + window, template_ind

def _remove_outliers(axis: np.ndarray, ind: np.ndarray, template_ind: np.ndarray, residual_threshold_min, residual_threshold_max):
    # Remove outliers using ransac-like robust inlier detection
    votes = _robust_inlier_detection(axis[template_ind], axis[template_ind] - axis[ind], residual_threshold_min, residual_threshold_max)
    inlier_mask = votes >= 0.5
    return ind[inlier_mask], template_ind[inlier_mask]

def warp(data: np.ndarray, reference: np.ndarray, mz: np.ndarray, nb_segments: int, window: int, factor: float, outlier_detection: bool, residual_threshold_max: float=0.1, residual_threshold_min: float=0.3) -> np.ndarray:
    """ Perform coarse alignment of a mass spectrum to a reference mass spectrum.

    Args:
        data (np.ndarray): A mass spectrum to be warped.
        reference (np.ndarray): The reference mass spectrum.
        mz (np.ndarray): The common m/z array between the mass spectra.
        nb_segments (int): The number of segments the data is split into before identifying the largest peak in the data.
        window (int): The window around the highest peak used to identify the shift which maximizes the dot product between the reference and the data around the maximal peak.
        factor (float): Window x factor determines the amount of shifts are considered to determine the shift which maximizes the dot product.
        outlier_detection (bool): Whether to perform outlier correction.
        residual_threshold_max (float): The maximal threshold value of the outlier detection algorithm. Defaults to 0.3.
        residual_threshold_min (float): The minimal threshold value of the outlier detection algorithm. Defaults to 0.1.

    Returns:
        np.ndarray: Returns the mass spectrum aligned to the reference mass spectrum.
    """
    ind, template_ind = _get_matches(reference, data, nb_segments, window, factor)
    
    if outlier_detection:
        ind, template_ind = _remove_outliers(mz, ind, template_ind, residual_threshold_min, residual_threshold_max)
    
        if ind.shape[0] <= 2:
            # If only 2 inliers are detected, the majority of the points are outliers. Hence we cannot trust the alignment and return the original data.
            return data
    actual_mz = cubic_spline(mz[template_ind].astype(np.float64), mz[ind].astype(np.float64), mz.astype(np.float64))
    return binning(mz, actual_mz.astype(mz.dtype), data)

@nb.njit(cache=True, inline='always', fastmath=fastmath)
def _b_func(x: np.ndarray, data: np.ndarray, mz: np.ndarray):
    result = np.zeros_like(x)
    
    idx1 = 0
    while idx1 != x.shape[0] and x[idx1] < mz[0]:
        idx1 += 1
    idx2 = x.shape[0] - 1
    while idx2 != -1 and x[idx2] > mz[-1]:
        idx2 -= 1
        
    index = searchsorted_merge(mz, x[idx1: idx2 + 1])
    w = (mz[index] - x[idx1: idx2 + 1]) / (mz[index] - mz[index - 1])
    result[idx1: idx2 + 1] = np.multiply(w, data[index - 1]) + np.multiply(1 - w, data[index])
    return result

@nb.njit([f'(i4[:, :])(f{ii}[:, :], f{ii}[:])' for ii in (8, 4)], cache=True, fastmath=fastmath, inline='always')
def _get_window_spline_nonzero(array: np.ndarray, y: np.ndarray) -> np.ndarray:
    # The result contains the indices of the first and last time array contains a value larger than machine precision / 2 in absolute value. The gradient is only calculated in this range.
    result = np.empty((array.shape[0], 2), dtype=np.int32)
    for i in range(array.shape[0]):
        idxs = np.flatnonzero(np.abs(array[i, :]) > np.finfo(y.dtype).eps / 2)
        result[i, 0] = idxs[0]
        result[i, 1] = idxs[-1]
    return result

@nb.njit([f'Tuple((f{ii}[:, :], i4[:, :]))(f{ii}[:], f{ii}[:])' for ii in (8, 4)], cache=True, fastmath=fastmath, inline='always')
def _jac_cs(mz: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    # Compute the Jacobian of the cubic spline function wrt to y values of the knots evaluated at the m/z values. Also returns the range of the Jacobian (per knot) where the values are larger than machine precision / 2 in absolute value for more efficient gradient calculations.
    jac = np.empty((nodes.shape[0], mz.shape[0]), dtype=mz.dtype)
    y = np.zeros(nodes.shape, dtype=np.float64)
    for i in range(jac.shape[0]):
        y[i] = 1
        jac[i, :] = cubic_spline(nodes, y, mz.astype(np.float64))
        y[i] = 0
    nonzero = _get_window_spline_nonzero(jac, y)
    return jac, nonzero

@nb.njit([f'Tuple((f{ii}, f{ii}[:]))(f{jj}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:], f{ii}[:, :], i4[:, :])' for ii in (8, 4) for jj in (8, 4)], cache=True, fastmath=fastmath, inline='always')
def _f_and_jac(S: np.ndarray, reference: np.ndarray, data: np.ndarray, mz: np.ndarray, nodes: np.ndarray, jac_cubic_spline: np.ndarray, nonzero: np.ndarray):
    # Compute the cost function and the gradient of the cost function for the optimization problem.
    x = cubic_spline(nodes, S.astype(reference.dtype), mz).astype(reference.dtype)
    
    jac_b = np.zeros_like(x)
    b = np.zeros_like(x)
    
    idx1 = 0
    while idx1 != x.shape[0] and x[idx1] < mz[0]:
        idx1 += 1
    idx2 = x.shape[0] - 1
    while idx2 != -1 and x[idx2] > mz[-1]:
        idx2 -= 1
    
    index = searchsorted_merge(mz, x[idx1: idx2 + 1])
    w = -1 / (mz[index] - mz[index - 1])
    jac_b[idx1: idx2 + 1] = np.multiply(w, data[index - 1]) + np.multiply(-w, data[index])
    # linear interpolation weights
    w *= (x[idx1: idx2 + 1] - mz[index])
    b[idx1: idx2 + 1] = np.multiply(w, data[index - 1]) + np.multiply(1 - w, data[index])
    
    norm_b = np.linalg.norm(b[idx1: idx2 + 1])
    reference = np.ascontiguousarray(reference)
    
    temp = - np.dot(reference[idx1: idx2 + 1], b[idx1: idx2 + 1] / norm_b)
    
    rhs = (reference + temp * b / norm_b) / norm_b
    grad = np.empty((jac_cubic_spline.shape[0],), jac_cubic_spline.dtype)
    for i in range(grad.shape[0]):
        grad[i] = - np.multiply(jac_b[nonzero[i, 0]: nonzero[i, 1] + 1], jac_cubic_spline[i, nonzero[i, 0]: nonzero[i, 1] + 1]) @ rhs[nonzero[i, 0]: nonzero[i, 1] + 1]
    
    return temp, grad

def _optimization(data: np.ndarray, reference: np.ndarray, mz: np.ndarray, template_ind: np.ndarray, ind: np.ndarray, delta: float=0.04, only_opt: bool=False):
    # Optimize the initial estimate of the warping knots using a mathematical optimization problem.
    S0 = np.hstack([mz[0], mz[template_ind], mz[-1]])
    if only_opt:
        nodes = np.copy(S0)
    else:
        nodes = cubic_spline(mz[template_ind], mz[ind], np.hstack([mz[0], mz[template_ind], mz[-1]]))
    jac_cubic_spline, nonzero = _jac_cs(mz, nodes)
    bounds = np.stack([S0 - delta, S0 + delta], axis=1)
    
    x = fmin_l_bfgs_b(_f_and_jac, S0, args=(reference, data / np.linalg.norm(data), mz, nodes, jac_cubic_spline, nonzero), bounds=bounds)
    
    return x[0], nodes.astype(np.float64)

def warp_optimization(data: np.ndarray, reference: np.ndarray, mz: np.ndarray, nb_segments: int, window: int, factor: float, outlier_detection: bool, delta: float=0.04, residual_threshold_max: float=0.1, residual_threshold_min: float=0.3, only_opt: bool=False):
    """ Perform fine alignment of a mass spectrum to a reference mass spectrum by solving a mathematical optimization problem.

    Args:
        data (np.ndarray): A mass spectrum to be warped.
        reference (np.ndarray): The reference mass spectrum.
        mz (np.ndarray): The common m/z array between the mass spectra.
        nb_segments (int): The number of segments the data is split into before identifying the largest peak in the data.
        window (int): The window around the highest peak used to identify the shift which maximizes the dot product between the reference and the data around the maximal peak.
        factor (float): Window x factor determines the amount of shifts are considered to determine the shift which maximizes the dot product.
        outlier_detection (bool): Whether to perform outlier correction.
        delta (float, optional): The bounds around the initial guess of the coarse alignment. Defaults to 0.04 Da.
        residual_threshold_max (float): The maximal threshold value of the outlier detection algorithm. Defaults to 0.3.
        residual_threshold_min (float): The minimal threshold value of the outlier detection algorithm. Defaults to 0.1.

    Returns:
        np.ndarray: Returns the mass spectrum aligned to the reference mass spectrum.
    """
    ind, template_ind = _get_matches(reference, data, nb_segments, window, factor)
    
    if outlier_detection and not only_opt:
        ind, template_ind = _remove_outliers(mz, ind, template_ind, residual_threshold_min, residual_threshold_max)
        
        if ind.shape[0] <= 2:
            # If only 2 inliers are detected, this means there was a lot of noise: return the orginal mz vector
            return data
    
    mz_ind, mz_template_ind = _optimization(data, reference, mz, template_ind, ind, delta, only_opt)
    actual_mz = cubic_spline(mz_template_ind, mz_ind, mz.astype(np.float64))
    return _b_func(actual_mz, data, mz)

def compute_warping_nodes(data: np.ndarray, reference: np.ndarray, mz: np.ndarray, nb_segments: int, window: int, factor: float, outlier_detection: bool, residual_threshold_max: float=0.1, residual_threshold_min: float=0.3):
    
    ind, template_ind = _get_matches(reference, data, nb_segments, window, factor)
    
    if outlier_detection:
        ind, template_ind = _remove_outliers(mz, ind, template_ind, residual_threshold_min, residual_threshold_max)
    
    return mz[ind], mz[template_ind]

def compute_warping_nodes_optimization(data: np.ndarray, reference: np.ndarray, mz: np.ndarray, nb_segments: int, window: int, factor: float, outlier_detection: bool, delta: float=0.04, residual_threshold_max: float=0.1, residual_threshold_min: float=0.3, only_opt: bool=False):

    ind, template_ind = _get_matches(reference, data, nb_segments, window, factor)
    
    if outlier_detection and not only_opt:
        ind, template_ind = _remove_outliers(mz, ind, template_ind, residual_threshold_min, residual_threshold_max)
        
        if ind.shape[0] <= 2:
            # If only 2 inliers are detected, this means there was a lot of noise: return the orginal mz vector
            return mz[template_ind], mz[ind]
    
    mz_ind, mz_template_ind = _optimization(data, reference, mz, template_ind, ind, delta, only_opt)
    return mz_template_ind, mz_ind

def transform(data: np.ndarray, mz: np.ndarray, nodes_x: np.ndarray, nodes_y: np.ndarray) -> np.ndarray:
    """ Apply given warping functions to a mass spectrum. """
    if nodes_x.shape[0] <= 2:
        # If only 2 nodes are given, return the original data
        return data
    actual_mz = cubic_spline(nodes_x.astype(np.float64), nodes_y.astype(np.float64), mz.astype(np.float64))
    return _b_func(actual_mz, data, mz)

class Alignment:
    
    def __init__(self, data: np.ndarray, mz: np.ndarray, reference: np.ndarray=None, nb_cores: int=1, instrument: str='tof', residual_threshold_max: float=None, residual_threshold_min: float=None, precompile: bool=True):
        """ Initializes an instance to perform alignment and passes the m/z array and intensity values (as a matrix).

        Args:
            data (np.ndarray): Matrix of the intensity values per pixel.
            mz (np.ndarray): Vector of the m/z array.
            reference (np.ndarray, optional): A given reference spectrum, for instance if you want to align to a specific spectrum. If 'None' it defaults to the average pixel after TIC normalization.
            nb_cores (int, optional): The number of cores used for the computation of the alignment procedure as well as performance metrics. Defaults to 1.
            instrument (str, optional): The type of instrument used, which determines default values for 'residual_threshold_max' and 'residual_threshold_min'. Defaults to 'tof'.
            residual_threshold_max (float, optional): The maximal threshold value of the outlier detection algorithm. It has predefined values for instrument='tof'|'orbitrap' if it defaults to None.
            residual_threshold_min (float, optional): The minimal threshold value of the outlier detection algorithm. It has predefined values for instrument='tof'|'orbitrap' if it defaults to None.
            precompile (bool, optional): Whether to precompile the Numba functions. Defaults to True.
        """
        self.dtype = data.dtype
        if instrument == 'tof':
            self.data = data
            self.mz = mz
            self.residual_threshold_max = 0.3 if not residual_threshold_max else residual_threshold_max
            self.residual_threshold_min = 0.1 if not residual_threshold_min else residual_threshold_min
        else:
            self.data = np.float64(data)
            self.mz = np.float64(mz)
            self.residual_threshold_max = 0.03 if not residual_threshold_max else residual_threshold_max
            self.residual_threshold_min = 0.01 if not residual_threshold_min else residual_threshold_min
        self._reference = reference if reference is not None else average_pixel(self.data)
        self._reference /= np.linalg.norm(self._reference)
        self.nb_cores = nb_cores
        self.instrument = instrument
        self.precompile = precompile

    def limit_mz_range(self, start_mz: float=None, end_mz: float=None):
        """ Limits the m/z range of the data, the m/z array and the reference from 'start_mz' to 'end_mz'.

        Args:
            start_mz (float, optional): Start of the chosen m/z range, if None: the smallest m/z value is chosen. Defaults to None.
            end_mz (float, optional): End of the chosen m/z range, if None: the largest m/z value is chosen. Defaults to None.
        """
        start_index = np.max([bisect_left(self.mz, start_mz) - 1, 0]) if start_mz else 0
        if end_mz:
            stop_index = np.min([bisect_right(self.mz, end_mz), self.mz.shape[0] - 1])
            self.data = self.data[:, start_index: stop_index + 1]
            self.mz = self.mz[start_index: stop_index + 1]
        else:
            self.data = self.data[:, start_index: ]
            self.mz = self.mz[start_index: ]
        
        if self._reference.shape[0] != self.mz.shape[0]:
            self._reference = self._reference[start_index: ] if not stop_index else self._reference[start_index: stop_index + 1]
            self._reference /= np.linalg.norm(self._reference)

    def align(self, nb_segments: int, window: int, factor: float, outlier_detection: bool):
        """ Perform coarse alignment of the data to the reference spectrum. Make sure that factor x window >= mz.shape[0] / nb_segments / 2.

        Args:
            nb_segments (int): The number of segments the data is split into before identifying the largest peak in the data.
            window (int): The window around the highest peak used to identify the shift which maximizes the dot product between the reference and the data around the maximal peak.
            factor (float): Window x factor determines the amount of shifts are considered to determine the shift which maximizes the dot product.
            outlier_detection (bool): Whether to perform outlier correction.

        Returns:
            tuple(np.ndarray, np.ndarray): Returns the aligned data and the corresponding m/z array.
        """
        if self.nb_cores == 1:
            if self.precompile:
                compile_numba(warp, self.data[0, :], self._reference, self.mz, nb_segments, window, factor, outlier_detection, self.residual_threshold_max, self.residual_threshold_min)
            s = time.perf_counter_ns()
            for i in tqdm(range(self.data.shape[0])):
                self.data[i, :] = warp(self.data[i, :], self._reference, self.mz, nb_segments, window, factor, outlier_detection, self.residual_threshold_max, self.residual_threshold_min)
        else:
            s = execute_parallel_along_axis(self.data, warp, args=(self._reference, self.mz, nb_segments, window, factor, outlier_detection, self.residual_threshold_max, self.residual_threshold_min), nb_cores=self.nb_cores, precompile=self.precompile)
        print('The data was warped in', np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2), 'seconds.')
            
        return self.data, self.mz if self.instrument == 'tof' else self.data.astype(self.dtype), self.mz.astype(self.dtype)
    
    def align_optimization(self, nb_segments: int, window: int, factor: float, outlier_detection: bool, delta: float=0.04, only_opt: bool=False):
        """ Perform fine alignment of the data to the reference spectrum by solving a mathematical optimization problem. Make sure that factor x window >= mz.shape[0] / nb_segments / 2.

        Args:
            nb_segments (int): The number of segments the data is split into before identifying the largest peak in the data.
            window (int): The window around the highest peak used to identify the shift which maximizes the dot product between the reference and the data around the maximal peak.
            factor (float): Window x factor determines the amount of shifts are considered to determine the shift which maximizes the dot product.
            outlier_detection (bool): Whether to perform outlier correction.
            delta (float, optional): The bounds around the initial guess of the coarse alignment. Defaults to 0.04 Da.
            only_opt (bool, optional): Whether to perform coarse alignment before fine alignment

        Returns:
            tuple(np.ndarray, np.ndarray): Returns the aligned data and the corresponding m/z array.
        """
        if self.nb_cores == 1:
            if self.precompile:
                compile_numba(warp_optimization, self.data[0, :], self._reference, self.mz, nb_segments, window, factor, outlier_detection, delta, self.residual_threshold_max, self.residual_threshold_min, only_opt)
            s = time.perf_counter_ns()
            for i in tqdm(range(self.data.shape[0])):
                self.data[i, :] = warp_optimization(self.data[i, :], self._reference, self.mz, nb_segments, window, factor, outlier_detection, delta, self.residual_threshold_max, self.residual_threshold_min, only_opt)
        else:
            s = execute_parallel_along_axis(self.data, warp_optimization, args=(self._reference, self.mz, nb_segments, window, factor, outlier_detection, delta, self.residual_threshold_max, self.residual_threshold_min), nb_cores=self.nb_cores, precompile=self.precompile)
        print('The data was warped in', np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2), 'seconds.')
            
        return self.data, self.mz if self.instrument == 'tof' else self.data.astype(self.dtype), self.mz.astype(self.dtype)
    
    def get_mass_dispersion(self, distance: int=10, nb_of_peaks: int=100, width: int=None):
        """ Compute the average and median mass dispersion of the data, and the cosine similarity of the data to the average pixel (after TIC normalization) of the data.

        Args:
            distance (int, optional): The distance between peaks, see 'scipy.signal.find_peaks'. Defaults to 10.
            nb_of_peaks (int, optional): The largest number of peaks to consider for computing the mass dispersion. Defaults to 100.
            width (int, optional): The width of the peaks, see 'scipy.signal.find_peaks'. Defaults to None.

        Returns:
            tuple(float): Returns the average, median mass dispersion of the data and the cosine similarity.
        """
        if self.nb_cores == 1:
            result = compute_mass_dispersion(self.data, self.mz, distance, nb_of_peaks, width)
        else:
            with ProcessPoolExecutor(max_workers=self.nb_cores) as executor:
                mass_dispersion = compute_mass_dispersion(self.data, self.mz, distance, nb_of_peaks, width, executor=executor)
            result = mass_dispersion
        print(f'\nMass dispersion [ppm]:\n\tAverage:\t\t{np.format_float_positional(result[0], precision=2)}\n\tMedian:\t\t\t{np.format_float_positional(result[1], precision=2)}\n\tCosine similarity:\t{np.format_float_positional(result[2], precision=4)}\n')
        return result
    
    def get_mass_dispersion_data(self, data: np.ndarray, mz: np.ndarray, distance: int=10, nb_of_peaks: int=100, width: int=None, verbose: bool=False, executor=None):
        """ Compute the average and median mass dispersion of the data, and the cosine similarity of the data to the average pixel (after TIC normalization) of the data.

        Args:
            distance (int, optional): The distance between peaks, see 'scipy.signal.find_peaks'. Defaults to 10.
            nb_of_peaks (int, optional): The largest number of peaks to consider for computing the mass dispersion. Defaults to 100.
            width (int, optional): The width of the peaks, see 'scipy.signal.find_peaks'. Defaults to None.

        Returns:
            tuple(float): Returns the average, median mass dispersion of the data and the cosine similarity.
        """
        if self.nb_cores == 1:
            result = compute_mass_dispersion(data, mz, distance, nb_of_peaks, width)
        else:
            mass_dispersion = compute_mass_dispersion(data, mz, distance, nb_of_peaks, width, executor)
            result = mass_dispersion
        if verbose:
            print(f'\nMass dispersion [ppm]:\n\tAverage:\t\t{np.format_float_positional(result[0], precision=2)}\n\tMedian:\t\t\t{np.format_float_positional(result[1], precision=2)}\nCosine similarity:\t\t{np.format_float_positional(result[2], precision=4)}\n')
        return result

    def get_best_params(self, nb_segments: list[int], window: list[int], factor: list[float], nb_of_spectra: int=1000, verbose: bool=False, distance: int=10, nb_of_peaks: int=100, width: int=None, metric: int=0) -> tuple:
        """ Perform a grid search on a subset of the data to find the best parameters for the alignment procedure with respect to the cosine similarity. The parameters considered are zip(nb_segments, window, factor). If nb_segments is less than 5, outlier detection is not performed. If nb_segments is larger than 5, outlier detection is performed. If nb_segments is equal to 5, alignment is performed with and without outlier detection. """
        data = self.data[np.random.randint(self.data.shape[0], size=(nb_of_spectra,)), :]
        d = np.copy(data)
        if self.nb_cores > 1:
            executor = ProcessPoolExecutor(max_workers=self.nb_cores)
            
        if verbose:
            print(f'Grid search on {nb_of_spectra} mass spectra of the data:\n')
            result = self.get_mass_dispersion_data(d, self.mz, distance, nb_of_peaks, width, False, None if self.nb_cores == 1 else executor)
            print(f'Metric: \t{np.format_float_positional(result[0], 4)}\t{np.format_float_positional(result[1], 4)}\t{np.format_float_positional(result[2], 4)}')
        
        result = []
        params = []
        # for nb_seg, win, fac in zip(nb_segments, window, factor):
        for nb_seg in nb_segments:
            for win in window:
                for fac in factor:
                    if fac * win >= d.shape[1] / nb_seg / 2:
                        break
                    if nb_seg <= 5:
                        if verbose:
                            print(f'nb_segments: {nb_seg}, window: {win}, factor: {fac}, outlier_detection: {False}')
                        d = np.copy(data)
                        if self.nb_cores == 1:
                            for i in tqdm(range(d.shape[0])):
                                d[i, :] = warp(d[i, :], self._reference, self.mz, nb_seg, win, fac, False, self.residual_threshold_max, self.residual_threshold_min)
                        else:
                            execute_parallel_along_axis_pool(d, warp, args=(self._reference, self.mz, nb_seg, win, fac, False, self.residual_threshold_max, self.residual_threshold_min), executor=executor, tqdm_args={'disable': True})
                        
                        result.append(self.get_mass_dispersion_data(d, self.mz, distance, nb_of_peaks, width, False, None if self.nb_cores == 1 else executor))
                        params.append((nb_seg, win, fac, False))
                        
                        if verbose:
                            print(f'Metric: \t{np.format_float_positional(result[-1][0], 4)}\t{np.format_float_positional(result[-1][1], 4)}\t{np.format_float_positional(result[-1][2], 4)}')
                    if nb_seg >= 5:
                        if verbose:
                            print(f'nb_segments: {nb_seg}, window: {win}, factor: {fac}, outlier_detection: {True}')
                        d = np.copy(data)
                        if self.nb_cores == 1:
                            for i in tqdm(range(d.shape[0])):
                                d[i, :] = warp(d[i, :], self._reference, self.mz, nb_seg, win, fac, True, self.residual_threshold_max, self.residual_threshold_min)
                        else:
                            execute_parallel_along_axis_pool(d, warp, args=(self._reference, self.mz, nb_seg, win, fac, True, self.residual_threshold_max, self.residual_threshold_min), executor=executor, tqdm_args={'disable': True})
                        
                        result.append(self.get_mass_dispersion_data(d, self.mz, distance, nb_of_peaks, width, False, None if self.nb_cores == 1 else executor))
                        params.append((nb_seg, win, fac, True))
                        
                        if verbose:
                            print(f'Metric: \t{np.format_float_positional(result[-1][0], 4)}\t{np.format_float_positional(result[-1][1], 4)}\t{np.format_float_positional(result[-1][2], 4)}')
        
        if self.nb_cores > 1:
            executor.shutdown(wait=True)
        result = np.array(result)
        result[:, 2] *= -1
        return params[np.argmin(result[:, metric])]
    
    def compute_warping_functions(self, nb_segments: int, window: int, factor: float, outlier_detection: bool):
        x = [None for _ in range(self.data.shape[0])]
        y = [None for _ in range(self.data.shape[0])]
        if self.nb_cores == 1:
            if self.precompile:
                compile_numba(compute_warping_nodes, self.data[0, :], self._reference, self.mz, nb_segments, window, factor, outlier_detection, self.residual_threshold_max, self.residual_threshold_min)
            s = time.perf_counter_ns()
            for i in tqdm(range(self.data.shape[0])):
                x[i], y[i] = compute_warping_nodes(self.data[i, :], self._reference, self.mz, nb_segments, window, factor, outlier_detection, self.residual_threshold_max, self.residual_threshold_min)
        else:
            s = execute_parallel_along_axis_lists(x, y, self.data, compute_warping_nodes, args=(self._reference, self.mz, nb_segments, window, factor, outlier_detection, self.residual_threshold_max, self.residual_threshold_min), nb_cores=self.nb_cores, precompile=self.precompile)
        print('The warping functions were computed in', np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2), 'seconds.')
            
        return x, y
    
    def compute_warping_functions_optimization(self, nb_segments: int, window: int, factor: float, outlier_detection: bool, delta: float=0.04, only_opt: bool=False):
        x = [None for _ in range(self.data.shape[0])]
        y = [None for _ in range(self.data.shape[0])]
        if self.nb_cores == 1:
            if self.precompile:
                compile_numba(compute_warping_nodes_optimization, self.data[0, :], self._reference, self.mz, nb_segments, window, factor, outlier_detection, delta, self.residual_threshold_max, self.residual_threshold_min, only_opt)
            s = time.perf_counter_ns()
            for i in tqdm(range(self.data.shape[0])):
                x[i], y[i] = compute_warping_nodes_optimization(self.data[i, :], self._reference, self.mz, nb_segments, window, factor, outlier_detection, delta, self.residual_threshold_max, self.residual_threshold_min, only_opt)
        else:
            s = execute_parallel_along_axis_lists(x, y, self.data, compute_warping_nodes_optimization, args=(self._reference, self.mz, nb_segments, window, factor, outlier_detection, delta, self.residual_threshold_max, self.residual_threshold_min), nb_cores=self.nb_cores, precompile=self.precompile)
        print('The warping functions were computed in', np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2), 'seconds.')
            
        return x, y
    
    def apply_warping_functions_to_data(self, x: list[np.ndarray], y: list[np.ndarray]):
        if self.nb_cores == 1:
            s = time.perf_counter_ns()
            for i in tqdm(range(self.data.shape[0])):
                self.data[i, :] = transform(self.data[i, :], self.mz, x[i], y[i])
        else:
            s = execute_parallel_along_axis(self.data, transform, args=[(self.mz, x[i], y[i]) for i in range(self.data.shape[0])], nb_cores=self.nb_cores)
        print('The data was warped in', np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2), 'seconds.')
            
        return self.data if self.instrument == 'tof' else self.data.astype(self.dtype)
    
    def apply_warping_functions_to_mz(self, mzs: list[np.ndarray], x: list[np.ndarray], y: list[np.ndarray]):
        result = [None for _ in range(len(mzs))]
        s = time.perf_counter_ns()
        for i in tqdm(range(len(mzs))):
            result[i] = cubic_spline(y[i].astype(np.float64), x[i].astype(np.float64), mzs[i].astype(np.float64))
        print('The m/z vectors were warped in', np.format_float_positional((time.perf_counter_ns() - s) / 1e9, precision=2), 'seconds.')

        return result
    
class CrossModalAlignment:
    def __init__(self, data: np.ndarray, mz_data: np.ndarray, reference: np.ndarray, mz_reference: np.ndarray, instrument: str='tof', residual_threshold_max: float=None, residual_threshold_min: float=None):
        """ Initializes an instance to perform cross modal alignment and passes the m/z array and intensity values (as a matrix).

        Args:
            data (np.ndarray): Matrix of the intensity values per pixel.
            mz (np.ndarray): Vector of the m/z array.
            reference (np.ndarray): A given reference spectrum, for instance if you want to align to a specific spectrum. If 'None' it defaults to the average pixel after TIC normalization.
            instrument (str, optional): The type of instrument used to measure data, which determines default values for 'residual_threshold_max' and 'residual_threshold_min'. Defaults to 'tof'.
            residual_threshold_max (float, optional): The maximal threshold value of the outlier detection algorithm. It has predefined values for instrument='tof' if it defaults to None.
            residual_threshold_min (float, optional): The minimal threshold value of the outlier detection algorithm. It has predefined values for instrument='tof' if it defaults to None.
        """
        self.dtype = data.dtype
        if instrument == 'tof':
            self.data = data
            self.mz_data = mz_data
            self.residual_threshold_max = 0.3 if not residual_threshold_max else residual_threshold_max
            self.residual_threshold_min = 0.1 if not residual_threshold_min else residual_threshold_min
        else:
            self.data = np.float64(data)
            self.mz_data = np.float64(mz_data)
            self.residual_threshold_max = 0.03 if not residual_threshold_max else residual_threshold_max
            self.residual_threshold_min = 0.01 if not residual_threshold_min else residual_threshold_min
        self.reference = reference 
        self.reference /= np.linalg.norm(self.reference)
        self.mz_reference = mz_reference
        self.reference_data = average_pixel(self.data)
        self.reference_data /= np.linalg.norm(self.reference_data)
        self.instrument = instrument
        
    def limit_mz_range(self, start_mz: float=None, end_mz: float=None):
        """ Limits the m/z range of the data, the m/z array, the reference and the reference m/z array from 'start_mz' to 'end_mz'.

        Args:
            start_mz (float, optional): Start of the chosen m/z range, if None: the smallest m/z value is chosen. Defaults to None.
            end_mz (float, optional): End of the chosen m/z range, if None: the largest m/z value is chosen. Defaults to None.
        """
        # Data
        start_index = np.max([bisect_left(self.mz_data, start_mz) - 1, 0]) if start_mz else 0
        if end_mz:
            stop_index = np.min([bisect_right(self.mz_data, end_mz), self.mz_data.shape[0] - 1])
            self.data = self.data[:, start_index: stop_index + 1]
            self.mz_data = self.mz_data[start_index: stop_index + 1]
            self.reference_data = self.reference_data[start_index: stop_index + 1]
        else:
            self.data = self.data[:, start_index: ]
            self.mz_data = self.mz_data[start_index: ]
            self.reference_data = self.reference_data[start_index: ]
            
        self.reference_data /= np.linalg.norm(self.reference_data)
            
        # Reference
        start_index = np.max([bisect_left(self.mz_reference, start_mz) - 1, 0]) if start_mz else 0
        if end_mz:
            stop_index = np.min([bisect_right(self.mz_reference, end_mz), self.mz_reference.shape[0] - 1])
            self.reference = self.reference[start_index: stop_index + 1]
            self.mz_reference = self.mz_reference[start_index: stop_index + 1]
        else:
            self.reference = self.reference[start_index: ]
            self.mz_reference = self.mz_reference[start_index: ]
        
        self.reference /= np.linalg.norm(self.reference)
        
    def align(self, nb_segments: int, window: int, factor: float, outlier_detection: bool) -> np.ndarray:
        """ Perform coarse alignment of the data to the reference spectrum.

        Args:
            nb_segments (int): The number of segments the data is split into before identifying the largest peak in the data.
            window (int): The window around the highest peak used to identify the shift which maximizes the dot product between the reference and the data around the maximal peak.
            factor (float): Window x factor determines the amount of shifts are considered to determine the shift which maximizes the dot product.
            outlier_detection (bool): Whether to perform outlier correction.

        Returns:
            tuple(np.ndarray, np.ndarray): Returns the aligned data and the corresponding m/z array.
        """
        self.reference = binning(self.mz_data, self.mz_reference, self.reference)
        self.reference /= np.linalg.norm(self.reference)
        
        ind, template_ind = _get_matches(self.reference, self.reference_data, nb_segments, window, factor)
    
        if outlier_detection:
            ind, template_ind = _remove_outliers(self.mz_data, ind, template_ind, self.residual_threshold_min, self.residual_threshold_max)
        
        actual_mz = cubic_spline(self.mz_data[template_ind], self.mz_data[ind], self.mz_data.astype(np.float64))
        result = binning_matrix(self.mz_data, actual_mz, self.data)
                
        return (result, self.mz_data) if self.instrument == 'tof' else (result.astype(self.dtype), self.mz_data.astype(self.dtype))
    
    def align_optimization(self, nb_segments: int, window: int, factor: float, outlier_detection: bool, delta: float=0.04) -> np.ndarray:
        """ Perform fine alignment of the data to the reference spectrum by solving a mathematical optimization problem.

        Args:
            nb_segments (int): The number of segments the data is split into before identifying the largest peak in the data.
            window (int): The window around the highest peak used to identify the shift which maximizes the dot product between the reference and the data around the maximal peak.
            factor (float): Window x factor determines the amount of shifts are considered to determine the shift which maximizes the dot product.
            outlier_detection (bool): Whether to perform outlier correction.
            delta (float, optional): The bounds around the initial guess of the coarse alignment. Defaults to 0.04 Da.

        Returns:
            tuple(np.ndarray, np.ndarray): Returns the aligned data and the corresponding m/z array.
        """
        self.reference = binning(self.mz_data, self.mz_reference, self.reference)
        self.reference /= np.linalg.norm(self.reference)
        
        ind, template_ind = _get_matches(self.reference, self.reference_data, nb_segments, window, factor)
    
        if outlier_detection:
            ind, template_ind = _remove_outliers(self.mz_data, ind, template_ind, self.residual_threshold_min, self.residual_threshold_max)
        
        mz_ind, mz_template_ind = _optimization(self.reference_data, self.reference, self.mz_data, template_ind, ind, delta)
        actual_mz = cubic_spline(mz_ind, mz_template_ind, self.mz_data.astype(np.float64))
        result = binning_matrix(self.mz_data, actual_mz, self.data)
                
        return (result, self.mz_data) if self.instrument == 'tof' else (result.astype(self.dtype), self.mz_data.astype(self.dtype))

