import numpy as np

from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor

from .utils import tic_normalize_and_average_pixel, l2_normalize, execute_indexed_parallel_executor

def get_ppm_std(mz_vector, data, w):
    # Get the standard deviation of the ppm error for a given peak, peak masses are computed using the centroiding algorithm
    data_bool = np.array([True for _ in range(data.shape[0])])
    result = []
    for i in range(data.shape[0]):
        window = data[i, :]
        peaks, _ = find_peaks(window)
        if peaks.shape[0] > 0:
            result.append(peaks[np.abs(peaks - w).argmin()])
        else:
            data_bool[i] = False
    avg = np.average(data, axis=0)
    ref_mz = parabolic_centroid(mz_vector, w.reshape((1,)), avg.reshape((1, avg.shape[0])))
    result = (parabolic_centroid(mz_vector, np.array(result), data[data_bool, :]) - ref_mz) / ref_mz * 1e6
    return np.std(result)

def parabolic_centroid(mz_vector, peak_indices, intensities):
    # Given the peak centers, and the surrounding m/z values and intensities, compute the peak mass using a parabolic fit        
    X = np.ndarray((peak_indices.shape[0], 3), dtype=intensities.dtype)
    Y = np.ndarray((peak_indices.shape[0], 3), dtype=intensities.dtype)
    
    for i in range(X.shape[0]):
        X[i, :] = mz_vector[peak_indices[i] - 1: peak_indices[i] + 2]
        Y[i, :] = intensities[i, peak_indices[i] - 1: peak_indices[i] + 2]
    
    a = ((Y[:, 2] - Y[:, 1]) / (X[:, 2] - X[:, 1]) - 
         (Y[:, 1] - Y[:, 0]) / (X[:, 1] - X[:, 0])) / (X[:, 2] - X[:, 0])
    
    ind = a != 0
    a = a[ind]
    X = X[ind]
    Y = Y[ind]
    
    b = ((Y[:, 2] - Y[:, 1]) / (X[:, 2] - X[:, 1]) * (X[:, 1] - X[:, 0]) + 
         (Y[:, 1] - Y[:, 0]) / (X[:, 1] - X[:, 0]) * (X[:, 2] - X[:, 1])) / (X[:, 2] - X[:, 0])             

    mzs_centers = ((1 / 2) * (-b + 2 * a * X[:, 1]) / a)
    
    return mzs_centers

def compute_mass_dispersion(data, mz_vector, distance=20, nb_of_peaks=100, width=1, executor: ProcessPoolExecutor=None):
    # Compute the mass dispersion of the sample of the nb_of_peaks most intense peaks in the average spectrum (after TIC normalization)
    
    normalized_data, reference = tic_normalize_and_average_pixel(data)
    
    peaks, properties = find_peaks(reference, distance=distance, height=np.average(reference), width=1)
    
    ordered_peaks = peaks[np.flip(np.argsort(properties['peak_heights']))][: nb_of_peaks]
    
    if width is None:
        width = properties['widths'][np.flip(np.argsort(properties['peak_heights']))][: nb_of_peaks]
    else:
        if hasattr(width, 'shape'):
            if width.shape[0] != nb_of_peaks:
                width = np.array([width[0] for _ in range(nb_of_peaks)])
        else:
            width = np.array([width for _ in range(nb_of_peaks)])
    width = np.array([int(np.ceil(w)) for w in width], dtype=np.int32)
    
    width = np.min(np.stack([width, ordered_peaks]), axis=0)
    
    ############################### centroid ##########################################                    
    ppm = dict()
    if executor:
        result = execute_indexed_parallel_executor(executor, get_ppm_std, args=[(mz_vector[ordered_peaks[i] - width[i]: ordered_peaks[i] + width[i] + 1], normalized_data[:, ordered_peaks[i] - width[i]: ordered_peaks[i] + width[i] + 1], width[i]) for i in range(ordered_peaks.shape[0])], tqdm_args={'disable': True})
        for i, d in result:
            ppm[ordered_peaks[i]] = d
    else:
        for peak, w in zip(ordered_peaks, width):
            ppm[peak] = get_ppm_std(mz_vector[peak - w: peak + w + 1], normalized_data[:, peak - w: peak + w + 1], w)

    average_ppm_std3 = np.mean(list(ppm.values()))
    median_ppm_std3 = np.median(list(ppm.values()))
    
    normalized_data = l2_normalize(data)
    reference /= np.linalg.norm(reference)
    cosine_similarity = normalized_data @ reference
    cosine_similarity = np.average(cosine_similarity)

    return average_ppm_std3, median_ppm_std3, cosine_similarity

