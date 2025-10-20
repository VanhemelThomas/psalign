import numpy as np
import xml.etree.ElementTree as ET
import time
import os

from collections import namedtuple
from tqdm import tqdm
from scipy.signal import find_peaks
from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.ImzMLParser import ImzMLParser

from .utils import searchsorted_merge, binning, interpolation


type_map = {
    "64-bit float": np.float64,
    "32-bit float": np.float32
}

def correction(signal):
    median = np.nanmedian(signal)
    signal -= median
    signal[signal < 0] = 0
    return signal
 
# iterating over directory and subdirectory to get desired result
def convert(path, dtype, low_mz=None, high_mz=None, distance=10, nb_of_peaks=1000):
    # distance and number of peaks are only used when every pixel has its own m/z array
    # giving file extensions
    ext = ('.imzML')
    if not os.path.exists(path):
        raise Exception(f'The file {path} does not exist.')
        
    # iterating over directory and subdirectory to get desired result
    path, name = os.path.split(path)
    if name.endswith(ext):
        fname = os.path.join(path, os.path.splitext(name)[0])
        start = time.time()
        print('.imzML and .ibd file = ', fname)
        
        # read .imzl file and extract pixels and spectra
        hupostr = '{http://psi.hupo.org/ms/mzml}'
        spectrumtup = namedtuple('spectrumtup', ['x', 'y', 'mzs', 'intensities'])
        datatup = namedtuple('datatup', ['length', 'encoded', 'offset'])
        
        def spectrum2dict(e):
            x = None
            y = None
            scanlist = e.find(hupostr + 'scanList')
            scan = scanlist.find(hupostr + 'scan')
            for cvpar in scan.iter(hupostr + 'cvParam'):
                if cvpar.attrib['name'] == 'position x':
                    x = int(cvpar.attrib['value'])
                if cvpar.attrib['name'] == 'position y':
                    y = int(cvpar.attrib['value'])
            bdlst = e.find(hupostr + 'binaryDataArrayList').findall(hupostr + 'binaryDataArray')
            mzselem, spectelem = bdlst  # fixme: not robust enough, mzarray not necessarily first
            for cvpar in mzselem.iter(hupostr + 'cvParam'):
                if cvpar.attrib['name'] == 'external array length':
                    mzlength = int(cvpar.attrib['value'])
                if cvpar.attrib['name'] == 'external encoded length':
                    mzencoded = int(cvpar.attrib['value'])
                if cvpar.attrib['name'] == 'external offset':
                    mzoffset = int(cvpar.attrib['value'])
            mzs = datatup(length=mzlength, encoded=mzencoded, offset=mzoffset)
            for cvpar in spectelem.iter(hupostr + 'cvParam'):
                if cvpar.attrib['name'] == 'external array length':
                    intlength = int(cvpar.attrib['value'])
                if cvpar.attrib['name'] == 'external encoded length':
                    intencoded = int(cvpar.attrib['value'])
                if cvpar.attrib['name'] == 'external offset':
                    intoffset = int(cvpar.attrib['value'])
            intensities = datatup(length=intlength, encoded=intencoded, offset=intoffset)
            return spectrumtup(x=x, y=y, mzs=mzs, intensities=intensities)
        
        def save_data(high_mz=None, low_mz=None):
            xmltree = ET.parse(fname + '.imzML')
            xmlroot = xmltree.getroot()
            
            for referenceableParamGroup in xmlroot.find("{http://psi.hupo.org/ms/mzml}referenceableParamGroupList"):
                if referenceableParamGroup.attrib['id'] == 'mzArray':
                    for cvparam in referenceableParamGroup:
                        if 'float' in cvparam.attrib['name']:
                            mz_dtype = type_map[cvparam.attrib['name']]
                elif referenceableParamGroup.attrib['id'] == 'intensities' or referenceableParamGroup.attrib['id'] == 'intensityArray':
                    for cvparam in referenceableParamGroup:
                        if 'float' in cvparam.attrib['name']:
                            intensities_dtype = type_map[cvparam.attrib['name']]
                            
            runkey = '{http://psi.hupo.org/ms/mzml}run'
            run = xmlroot.find(runkey)
            spectrumlistkey = '{http://psi.hupo.org/ms/mzml}spectrumList'
            spectrumlist = run.find(spectrumlistkey)
            spectrumkey = '{http://psi.hupo.org/ms/mzml}spectrum'
            spectraelems = spectrumlist.findall(spectrumkey)
        
            imzml_data = list(map(spectrum2dict, spectraelems))
            
            offset = -1
            single_mz = True
            for d in imzml_data:
                if offset == -1:
                    offset = d.mzs.offset
                if offset != d.mzs.offset:
                    single_mz = False
                    break
            
            if single_mz:
                mz_vector = np.memmap(filename=fname + '.ibd', dtype=mz_dtype, mode='r', offset=imzml_data[0].mzs.offset, shape=(imzml_data[0].mzs.length,)).astype(dtype)
                print(f'Measured spectra are between {np.format_float_positional(mz_vector[0], 3)} Da and {np.format_float_positional(mz_vector[-1], 3)} Da and in {mz_vector.shape[0]} m/z bins.')
            else:
                print('Every pixel has a separate m/z array.')
                mzs = list(map(lambda x: np.memmap(filename=fname + '.ibd', dtype=mz_dtype, mode='r', offset=x.mzs.offset, shape=(x.mzs.length,)), imzml_data))
                
                if high_mz is None:
                    high_mz = np.max(np.array(list(map(lambda x: np.max(x), mzs))))
                if low_mz is None:
                    low_mz = np.min(np.array(list(map(lambda x: np.min(x), mzs))))
                low_mz = np.max([high_mz, np.min(np.array(list(map(lambda x: np.min(x), mzs))))])
                high_mz = np.min([low_mz, np.max(np.array(list(map(lambda x: np.max(x), mzs))))])
                mz_vector = mzs[np.argmax(np.array(list(map(lambda x: np.argmin(np.abs(high_mz - x) - np.argmin(np.abs(low_mz - x))), mzs))))]
                
                diff = np.diff(mz_vector)
                min_ind = np.argmin(diff)
                max_ind = -(10 - np.argmin(diff[-10:]))
                print(diff[min_ind], diff[max_ind], mz_vector[min_ind], mz_vector[max_ind])
                a = (diff[max_ind] - diff[min_ind]) / (mz_vector[max_ind] - mz_vector[min_ind])
                b = - a * mz_vector[min_ind] + diff[min_ind]

                def delta(m):
                    return a * m + b
                
                print('Generating a common m/z array.')
                for mz in tqdm(mzs):
                    idxs = searchsorted_merge(mz_vector, mz)
                    idxs_more_than_delta = np.logical_and(mz_vector[idxs % mz_vector.shape[0]] - mz > delta(mz), np.abs(mz_vector[idxs - 1] - mz) > delta(mz))
                    if idxs_more_than_delta.sum() > 0:
                        mz_vector = np.insert(mz_vector, idxs[idxs_more_than_delta], mz[idxs_more_than_delta])
                            
                mz_bin = np.zeros_like(mz_vector)
                for m in mzs:
                    idxs = searchsorted_merge(mz_vector, m)
                    mz_bin[idxs % mz_vector.shape[0]] += 1
                    
                mz_vector = mz_vector.astype(dtype)
                
                avg = np.zeros_like(mz_vector)
                
                for i in tqdm(range(len(mzs))):
                    d = interpolation(mz_vector, mzs[i].astype(dtype), intensities[i])
                    avg += d / np.sum(d)
                avg /= np.sum(avg)
                
                peaks, properties = find_peaks(avg, distance=distance, height=np.average(avg), width=1)
                widths = properties['widths']
                
                idxs = np.sort(np.flip(np.argsort(properties['peak_heights']))[: nb_of_peaks])
                peaks = peaks[idxs]
                widths = widths[idxs]
                
                mz = []
                buffer = 3
                for w, p in zip(widths, peaks):
                    w = int(np.ceil(w))
                    mz.append(mz_vector[p - w - buffer: p + w + buffer + 1])
                    
                mz_vector = np.concatenate(mz)
                mz_vector = np.sort(np.unique(mz_vector))
                
            mz_vector = mz_vector.astype(dtype)
            
            intensities = list(map(lambda x: np.memmap(filename=fname + '.ibd', dtype=intensities_dtype, mode='c', offset=x.intensities.offset, shape=(x.intensities.length,)).astype(dtype), imzml_data))
            
            row2grid = np.array(list(map(lambda p: (p.x, p.y), imzml_data)), dtype=int)
                    
            data_matrix = np.empty((len(intensities), mz_vector.shape[0]), dtype=dtype)
        
            if single_mz:
                for i in range(data_matrix.shape[0]):
                    data_matrix[i, :] = intensities[i].astype(dtype)
            else:
                print('Binning all pixels to the common m/z array.')
                for i in tqdm(range(len(mzs))):
                    data_matrix[i, :] = interpolation(mz_vector, mzs[i].astype(dtype), intensities[i])
                
            for i in range(data_matrix.shape[0]):
                data_matrix[i, :] = correction(data_matrix[i, :])
                if np.isclose(np.sum(data_matrix[i, :]), 0, atol=1e-4):
                    data_matrix[i, :] = np.ones_like(data_matrix[i, :], dtype=data_matrix.dtype)
            
            print(f'Saving {data_matrix.shape[0]} mass spectra of {data_matrix.shape[1]} m/z values between {np.format_float_positional(mz_vector[0], 3)} Da and {np.format_float_positional(mz_vector[-1], 3)} Da, the m/z array and the pixel locations to {path}/{os.path.splitext(name)[0]}.npz')
            
            np.savez(f"{path}/{os.path.splitext(name)[0]}.npz", data=data_matrix, axis=mz_vector, location=row2grid)
                    
        save_data(high_mz=high_mz, low_mz=low_mz)
                
        # Uncomment to save memory (removes the imzML and ibd files after convertion)
        # os.remove(f'{path}/{name.replace(".imzML", ".ibd")}')
        # os.remove(f'{path}/{name}')
        end = time.time()
        timeVal = end - start
        print(f'Finished in {np.format_float_positional(timeVal, 3)} seconds.')

def convert_pyimzml(path, dtype, low_mz=None, high_mz=None, distance=10, nb_of_peaks=1000):
    # distance and number of peaks are only used when every pixel has its own m/z array
    # giving file extensions
    ext = ('.imzML')
    if not os.path.exists(path):
        raise Exception(f'The file {path} does not exist.')
        
    # iterating over directory and subdirectory to get desired result
    path, name = os.path.split(path)
    if name.endswith(ext):
        fname = os.path.join(path, os.path.splitext(name)[0])
        start = time.time()
        print('.imzML and .ibd file = ', fname)
        
        from pyimzml.ImzMLParser import ImzMLParser

        intensities = []
        mzs = []
        row2grid = []
        p = ImzMLParser(f'{path}/{name}')

        for idx, coords in enumerate(p.coordinates):
            mz, hs = p.getspectrum(idx)    
            intensities.append(hs)
            mzs.append(mz)
            row2grid.append(coords)
            
        single_mz = np.unique([len(m) for m in mzs]).shape[0] == 1
        
        if single_mz:
            mz_vector = mzs[0].astype(dtype)
            print(f'Measured spectra are between {np.format_float_positional(mz_vector[0], 3)} Da and {np.format_float_positional(mz_vector[-1], 3)} Da and in {mz_vector.shape[0]} m/z bins.')
        else:
            print('Every pixel has a separate m/z array.')
            if high_mz is None:
                high_mz = np.max(np.array(list(map(lambda x: np.max(x), mzs))))
            if low_mz is None:
                low_mz = np.min(np.array(list(map(lambda x: np.min(x), mzs))))
            low_mz = np.max([high_mz, np.min(np.array(list(map(lambda x: np.min(x), mzs))))])
            high_mz = np.min([low_mz, np.max(np.array(list(map(lambda x: np.max(x), mzs))))])
            mz_vector = mzs[np.argmax(np.array(list(map(lambda x: np.argmin(np.abs(high_mz - x) - np.argmin(np.abs(low_mz - x))), mzs))))]
            
            diff = np.diff(mz_vector)
            min_ind = np.argmin(diff)
            max_ind = -(10 - np.argmin(diff[-10:]))
            print(diff[min_ind], diff[max_ind], mz_vector[min_ind], mz_vector[max_ind])
            a = (diff[max_ind] - diff[min_ind]) / (mz_vector[max_ind] - mz_vector[min_ind])
            b = - a * mz_vector[min_ind] + diff[min_ind]

            def delta(m):
                return a * m + b
            
            print('Generating a common m/z array.')
            for mz in tqdm(mzs):
                idxs = searchsorted_merge(mz_vector, mz)
                idxs_more_than_delta = np.logical_and(mz_vector[idxs % mz_vector.shape[0]] - mz > delta(mz), np.abs(mz_vector[idxs - 1] - mz) > delta(mz))
                if idxs_more_than_delta.sum() > 0:
                    mz_vector = np.insert(mz_vector, idxs[idxs_more_than_delta], mz[idxs_more_than_delta])
                        
            mz_bin = np.zeros_like(mz_vector)
            for m in mzs:
                idxs = searchsorted_merge(mz_vector, m)
                mz_bin[idxs % mz_vector.shape[0]] += 1
                
            mz_vector = mz_vector.astype(dtype)
            
            avg = np.zeros_like(mz_vector)
            
            for i in tqdm(range(len(mzs))):
                d = interpolation(mz_vector, mzs[i].astype(dtype), intensities[i])
                avg += d / np.sum(d)
            avg /= np.sum(avg)
            
            peaks, properties = find_peaks(avg, distance=distance, height=np.average(avg), width=1)
            widths = properties['widths']
            
            idxs = np.sort(np.flip(np.argsort(properties['peak_heights']))[: nb_of_peaks])
            peaks = peaks[idxs]
            widths = widths[idxs]
            
            mz = []
            buffer = 3
            for w, p in zip(widths, peaks):
                w = int(np.ceil(w))
                mz.append(mz_vector[p - w - buffer: p + w + buffer + 1])
                
            mz_vector = np.concatenate(mz)
            mz_vector = np.sort(np.unique(mz_vector))
            
        data_matrix = np.empty((len(intensities), mz_vector.shape[0]), dtype=dtype)
        
        if single_mz:
            for i in range(data_matrix.shape[0]):
                data_matrix[i, :] = intensities[i].astype(dtype)
        else:
            print('Binning all pixels to the common m/z array.')
            for i in tqdm(range(len(mzs))):
                data_matrix[i, :] = interpolation(mz_vector, mzs[i].astype(dtype), intensities[i])
            
        for i in range(data_matrix.shape[0]):
            data_matrix[i, :] = correction(data_matrix[i, :])
            if np.isclose(np.sum(data_matrix[i, :]), 0, atol=1e-4):
                data_matrix[i, :] = np.ones_like(data_matrix[i, :], dtype=data_matrix.dtype)
        
        print(f'Saving {data_matrix.shape[0]} mass spectra of {data_matrix.shape[1]} m/z values between {np.format_float_positional(mz_vector[0], 3)} Da and {np.format_float_positional(mz_vector[-1], 3)} Da, the m/z array and the pixel locations to {path}/{os.path.splitext(name)[0]}.npz')
        
        np.savez(f"{path}/{os.path.splitext(name)[0]}.npz", data=data_matrix, axis=mz_vector, location=row2grid)
                
        # Uncomment to save memory (removes the imzML and ibd files after convertion)
        # os.remove(f'{path}/{name.replace(".imzML", ".ibd")}')
        # os.remove(f'{path}/{name}')
        end = time.time()
        timeVal = end - start
        print(f'Finished in {np.format_float_positional(timeVal, 3)} seconds.')
        
def save_to_imzml(path, data, mzs, locations, dtype=np.float32):
    with ImzMLWriter(path) as writer:
        for i in range(len(data)):
            writer.addSpectrum(mzs[i].astype(dtype), data[i].astype(dtype), coords=locations[i])
    
def read_imzml(path):
    with ImzMLParser(path) as parser:
        mzs = []
        data = []
        locations = []
        for i in range(len(parser.coordinates)):
            mz, intensity = parser.getspectrum(i)
            mzs.append(np.array(mz))
            data.append(np.array(intensity))
            locations.append(parser.coordinates[i])
    return data, mzs, locations

