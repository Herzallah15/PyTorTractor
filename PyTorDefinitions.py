import numpy as np
import torch
from typing import Optional
import numpy as np
import torch
from torch import nn
import platform
import warnings
import h5py
from itertools import product
from functools import reduce
from TorClasses import *
from contractions_handler import *
from Hadrontractions_Converter import *
from Hadron_Info_Converter import *

index_map = {
    (1,0,0): 'a',
    (1,0,1): 'b',
    (1,0,2): 'c',
    (1,1,0): 'd',
    (1,1,1): 'e',
    (1,1,2): 'f',
    (1,2,0): 'g',
    (1,2,1): 'h',
    (1,2,2): 'i',

    (0,0,0): 'j',
    (0,0,1): 'k',
    (0,0,2): 'l',
    (0,1,0): 'm',
    (0,1,1): 'n',
    (0,1,2): 'o',
    (0,2,0): 'p',
    (0,2,1): 'q',
    (0,2,2): 'r',

    (2,0,0): 's',
    (2,0,1): 't',
    (2,0,2): 'u',
    (2,1,0): 'v',
    (2,1,1): 'w',
    (2,2,2): 'x',

    (3,0,0): 'y',
    (3,0,1): 'z',
    (3,0,2): 'Z',
}

def ddir(path):
    if np.all(path == np.array([0,0,0])):
        return '0'
    else:
        raise ValueError('Displacement_CannotBe')
def momentum(string_value):
    if string_value == 'mom_ray_000':
        return 'px0_py0_pz0'
def hdrn_type(x):
    if x[0] == 'meson_operators':
        return 'M'
    elif x[1] == 'baryon_operators':
        return 'B'


def get_best_device(use_gpu: bool = True, device_id: Optional[int] = None, verbose: bool = True) -> torch.device:
    if not use_gpu:
        device = torch.device("cpu")
        if verbose:
            print(f"Using CPU (GPU usage disabled)")
        return device
    if torch.cuda.is_available():
        if device_id is not None:
            if device_id >= torch.cuda.device_count():
                warnings.warn(f"GPU device {device_id} not available. Using GPU 0 instead.")
                device_id = 0
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cuda")
        
        if verbose:
            gpu_name = torch.cuda.get_device_name(device)
            gpu_count = torch.cuda.device_count()
            memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"Using NVIDIA GPU: {gpu_name}")
            print(f"Device: {device} ({gpu_count} GPU(s) available)")
            print(f"Memory: {memory_gb:.1f} GB")
        return device
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.zeros(1, device='mps')
            del test_tensor
            device = torch.device("mps")
            if verbose:
                print(f"Using Apple Silicon GPU (MPS)")
                print(f"Device: {device}")
                print(f"System: {platform.machine()}")
            return device
        except Exception as e:
            if verbose:
                warnings.warn(f"MPS available but not functional: {e}. Falling back to CPU.")
    device = torch.device("cpu")
    if verbose:
        print(f"Using CPU")
        print(f"Reason: No compatible GPU found or GPU unavailable")
        print(f"System: {platform.system()} {platform.machine()}")
        system = platform.system().lower()
        if system in ['linux', 'windows']:
            print(f"For NVIDIA GPU support, install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        elif system == 'darwin':
            print(f"For Apple Silicon GPU support, ensure macOS 12.3+ and install: pip install torch torchvision torchaudio")
    return device