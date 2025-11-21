from PyTorTractor_SingleHadron import *
import h5py
import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np

#normalized_pattern = NormalizePattern(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}')
def NormalizePattern(Mode_Indices ,Ps_In, Ps_Out):
    ain = {}
    counter = 0
    M_idx = ''
    Ps_In_idx = ''
    Ps_Out_idx = ''
    for i in Mode_Indices:
        if i not in ain:
            ain[i] = chr(65 + counter)
            counter += 1
        M_idx += ain[i]
    for i in Ps_In:
        if i not in ain:
            ain[i] = chr(65 + counter)
            counter += 1
        Ps_In_idx += ain[i]
    for i in Ps_Out:
        if i not in ain:
            ain[i] = chr(65 + counter)
            counter += 1
        Ps_Out_idx += ain[i]
    return f'{M_idx},{Ps_In_idx}->{Ps_Out_idx}'






HDF5_FILE = "einsum_paths.h5"

def path_exists_in_hdf5(cache_key):
    with h5py.File(HDF5_FILE, 'a') as f:
        return cache_key in f
def load_path_from_hdf5(cache_key):
    with h5py.File(HDF5_FILE, 'r') as f:
        pickled = f[cache_key][()]
        return pickle.loads(pickled)
def save_path_to_hdf5(cache_key, path):
    with h5py.File(HDF5_FILE, 'a') as f:
        if cache_key in f:
            del f[cache_key]
        # Store as binary, not as vlen string
        pickled = pickle.dumps(path)
        f.create_dataset(cache_key, data=np.void(pickled))

'''
def load_path_from_hdf5(cache_key):
    with h5py.File(HDF5_FILE, 'r') as f:
        pickled = f[cache_key][()].tobytes()
        return pickle.loads(pickled)
def load_path_from_hdf5(cache_key):
    with h5py.File(HDF5_FILE, 'r') as f:
        return pickle.loads(f[cache_key][()])
def save_path_to_hdf5(cache_key, path):
    with h5py.File(HDF5_FILE, 'a') as f:
        if cache_key in f:
            del f[cache_key]
        f.create_dataset(cache_key, data=pickle.dumps(path), dtype=h5py.special_dtype(vlen=bytes))
'''