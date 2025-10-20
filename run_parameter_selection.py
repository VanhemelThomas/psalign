import numpy as np
import argparse
import multiprocessing as mp

from src.psalign.alignment import Alignment
from src.psalign.imzml import convert


if __name__ == '__main__':
    
    # & /miniconda3/envs/psalign/python.exe /psalign/run_alignment.py -p './data/<file>.imzML' -smz 100 -emz 1000 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='path to an .imzML file or a .npz file containing a "data" matrix containing the pixels as rows and the m/z values as columns and an "axis" vector containing the m/z array')
    parser.add_argument('-o', '--optimize', action='store_true', help='whether to optimize the alignment')
    parser.add_argument('-c', '--cores', type=int, required=False, default=mp.cpu_count() // 2, help='number of cores to use')
    parser.add_argument('-smz', '--start-mz', type=float, required=False, default=None, help='the start m/z value for the alignment')
    parser.add_argument('-emz', '--end-mz', type=float, required=False, default=None, help='the stop m/z value for the alignment')
    
    args = parser.parse_args()
    
    path = args.path
    nb_cores = args.cores
    start_mz = args.start_mz
    end_mz = args.end_mz
    
    nb_segments = [3, 5, 10, 20]
    window = [100, 200, 500, 1000, 2000]
    factor = [1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2]
    
    instrument = 'tof'
    reference = None
    
    if not path.endswith('.imzML') and not path.endswith('.npz'):
        raise Exception('The file is not a .imzML or .npz file.')
    else:
        if path.endswith('.imzML'):
            convert(path, np.float32)
            path = path.replace('.imzML', '.npz')
        file = np.load(path)
        data = file['data']
        mz = file['axis']
        del file
    
    alignment = Alignment(np.copy(data), mz, reference, nb_cores, instrument)
    alignment.limit_mz_range(start_mz, end_mz)
    
    nb_segments, window, factor, outlier_detection = alignment.get_best_params(nb_segments, window, factor, nb_of_spectra=1000, verbose=True)
    
    print(f'Best parameters:\n\tNumber of segments:\t{nb_segments}\n\tFactor:\t\t\t{factor}\n\tWindow:\t\t\t{window}\n\tOutlier detection:\t{outlier_detection}')
    
    alignment.get_mass_dispersion()
    alignment.align(nb_segments, window, factor, outlier_detection)
    print('Mass dispersion after alignment without optimization:')
    alignment.get_mass_dispersion()
    
    del alignment
    
    alignment = Alignment(data, mz, reference, nb_cores, instrument)
    alignment.limit_mz_range(start_mz, end_mz)
    
    alignment.align_optimization(nb_segments, window, factor, outlier_detection)
    print('Mass dispersion after alignment with optimization:')
    alignment.get_mass_dispersion()
    
    