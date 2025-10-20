import numpy as np
import numba as nb

fastmath = True

@nb.njit([f'f{ii}[:](f{ii}[:, ::1], f{ii}[::1])' for ii in (4, 8)], cache=True, fastmath=fastmath, inline='always')
def lr_fit(X: np.ndarray, y: np.ndarray):
    # Returns the solution of X @ x = y in the least squares sense
    return np.linalg.lstsq(X, y.astype(X.dtype))[0]

@nb.njit([f'f{ii}[:](f{ii}[:, ::1], f{ii}[::1])' for ii in (4, 8)], cache=True, fastmath=fastmath, inline='always')
def lr_predict(X: np.ndarray, A: np.ndarray):
    return X @ A

@nb.njit([f'f{ii}[:](f{ii}[:], f{ii}[:])' for ii in (4, 8)], cache=True, fastmath=fastmath, inline='always')
def ae(x: np.ndarray, y: np.ndarray):
    # returns the absolute error between x and y
    return np.abs(x - y)

@nb.njit([f'f{ii}(f{ii}[:], f{ii}[:])' for ii in (4, 8)], cache=True, fastmath=fastmath, inline='always')
def mae(x: np.ndarray, y: np.ndarray):
    # returns the mean absolute error between x and y
    return np.mean(ae(x, y))

@nb.njit([f'i8[:, :](i8, i8, i8)'], cache=True, fastmath=fastmath, inline='always')
def precompute_choice(n: int, k: int, nb_possible_choices: int):
    # Precompute all possible choices of k elements in n elements
    result = np.zeros((nb_possible_choices, n), dtype=np.int64)
    row = np.arange(k)
    index = 0
    result[index, row] = 1
    
    for index in range(1, nb_possible_choices):
        
        for i in range(k - 1, -1, -1):
            if row[i] < i + n - k:
                break
        
        row[i] += 1
        
        for j in range(i + 1, k):
            row[j] = row[j - 1] + 1
            
        result[index, row] = 1
            
    return result

@nb.njit([f'f{ii}(i8,)' for ii in (8,)], cache=True, fastmath=fastmath, inline='always')
def factorial(n):
    # returns the factorial of n
    return np.prod(np.arange(1, n + 1))

@nb.njit([f'i8(i8, i8)'], cache=True, fastmath=fastmath, inline='always')
def combinations_k_in_n(n, k):
    # returns the number of combinations of k elements in n elements
    return np.int64(np.round(factorial(n) / (factorial(n - k) * factorial(k))))

@nb.njit(cache=True, inline='always', fastmath=fastmath)
def compute_inliers(X: np.ndarray, y: np.ndarray, max_iter: int=100, residual_threshold: float=1, random_state: int=13):
    # Compute the inliers of the data using RANSAC. However, if the number of different possible choices is less than max_iter, generate all possible choices and loop over them instead of sampling them randomly.
    X = np.append(np.ones((X.shape[0], 1), dtype=X.dtype), X, axis=1)
    n = 2
    most_nb_inliers = 0
    best_error = np.inf
    # If number of different possible choices is less than max_iter, generate all possible choices and loop over them
    nb_possible_choices = combinations_k_in_n(X.shape[0], n)
    if nb_possible_choices <= max_iter:
        permutations = precompute_choice(X.shape[0], n, nb_possible_choices)
        for idxs in permutations:
            idxs = idxs == 1
            model = lr_fit(X[idxs], y[idxs])
            not_idxs = idxs == 0

            thresholded = ae(y[not_idxs], lr_predict(X[not_idxs], model)) < residual_threshold

            inlier_idxs = np.flatnonzero(not_idxs)[thresholded]
            idxs = np.flatnonzero(idxs)

            if inlier_idxs.shape[0] + idxs.shape[0] >= most_nb_inliers:
                inliers = np.append(idxs, inlier_idxs)
                model = lr_fit(X[inliers], y[inliers])
                error = mae(y[inliers], lr_predict(X[inliers], model))

                if inliers.shape[0] > most_nb_inliers or (error < best_error and inliers.shape[0] == most_nb_inliers):
                    best_inliers = inliers
                    most_nb_inliers = best_inliers.shape[0]
                    if most_nb_inliers == X.shape[0]:
                        return best_inliers
                    best_error = error
    else:
        # Perform RANSAC
        np.random.seed(random_state)
        for _ in range(max_iter):
            idxs = np.random.permutation(np.arange(X.shape[0], dtype=np.int64))
            
            inliers = idxs[: n]
            model = lr_fit(X[inliers], y[inliers])
            
            thresholded = ae(y[idxs][n:], lr_predict(X[idxs][n:], model)) < residual_threshold

            inlier_idxs = idxs[n:][np.flatnonzero(thresholded)]
            
            if inlier_idxs.shape[0] >= most_nb_inliers:
                inlier_points = np.append(inliers, inlier_idxs)
                better_model = lr_fit(X[inlier_points], y[inlier_points])
                error = mae(y[inlier_points], lr_predict(X[inlier_points], better_model))

                if inlier_idxs.shape[0] > most_nb_inliers or (error < best_error and inlier_idxs.shape[0] == most_nb_inliers):
                    best_inliers = inlier_points
                    most_nb_inliers = best_inliers.shape[0]
                    if most_nb_inliers == X.shape[0]:
                        return best_inliers
                    best_error = error
                
    return best_inliers

