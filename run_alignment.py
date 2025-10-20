import numpy as np
import argparse
import multiprocessing as mp

from src.psalign.alignment import Alignment
from src.psalign.imzml import convert


if __name__ == '__main__':
    
    # & /miniconda3/envs/psalign/python.exe /psalign/run_alignment.py -p './data/<file>.imzML' -o -c 8 -s 5 -w 2000 -f 1.03 -od -smz 100 -emz 1000 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='path to an .imzML file or a .npz file containing a "data" matrix containing the pixels as rows and the m/z values as columns and an "axis" vector containing the m/z array')
    parser.add_argument('-o', '--optimize', action='store_true', help='whether to optimize the alignment')
    parser.add_argument('-c', '--cores', type=int, required=False, default=mp.cpu_count() // 2, help='number of cores to use')
    parser.add_argument('-s', '--segments', type=int, required=False, default=5, help='number of segments to use for the alignment')
    parser.add_argument('-w', '--window', type=int, required=False, default=2000, help='window size to use for the alignment')
    parser.add_argument('-f', '--factor', type=float, required=False, default=1.05, help='factor to use for the alignment, together with the window size determines the maximal mass shift')
    parser.add_argument('-od', '--outlier-detection', action='store_true', help='whether to use outlier detection for the alignment')
    parser.add_argument('-smz', '--start-mz', type=float, required=False, default=None, help='the start m/z value for the alignment')
    parser.add_argument('-emz', '--end-mz', type=float, required=False, default=None, help='the stop m/z value for the alignment')
    
    args = parser.parse_args()
    
    path = args.path
    optimize = True if args.optimize else False
    nb_cores = args.cores
    nb_segments = args.segments
    window = args.window
    factor = args.factor
    outlier_detection = True if args.outlier_detection else False
    start_mz = args.start_mz
    end_mz = args.end_mz
    
    instrument = 'tof'
    reference = None
    
    print(f'Sample {path}:\n\tNumber of segments:\t{nb_segments}\n\tFactor:\t\t\t{factor}\n\tWindow:\t\t\t{window}\n\tOutlier detection:\t{outlier_detection}\n\tOptimize:\t\t{optimize}\n\tM/z range:\t\t{start_mz} - {end_mz} Da')
    
    if not path.endswith('.imzML') and not path.endswith('.npz'):
        raise Exception('The file is not a .imzML or .npz file.')
    else:
        if path.endswith('.imzML'):
            convert(path, np.float32)
            path = path.replace('.imzML', '.npz')
        file = np.load(path)
        data = file['data']
        mz = file['axis']
    
    alignment = Alignment(data, mz, reference, nb_cores, instrument)
    alignment.limit_mz_range(start_mz, end_mz)
    del data
    
    alignment.get_mass_dispersion()
    if optimize:
        alignment.align_optimization(nb_segments, window, factor, outlier_detection)
    else:
        alignment.align(nb_segments, window, factor, outlier_detection)
    alignment.get_mass_dispersion()
    
    