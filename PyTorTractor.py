import numpy as np
import torch
from torch import nn
import platform
import warnings
from typing import Optional
import h5py
from itertools import product
from functools import reduce
from TorClasses import *
from contractions_handler import *
from Hadrontractions_Converter import *
from Hadron_Info_Converter import *



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



class PyCorrTorch:
    def __init__(self, SinkTime = None, SourceTime = None, 
                 Hadrons = None, Path_Wicktract = None, 
                 Path_Perambulator = None, Path_ModeDoublet = None, Path_ModeTriplet = None, useGPU = True, Device_ID = None):
        if None in (SinkTime, SourceTime, Hadrons, Path_Wicktract, Path_Perambulator):
            raise ValueError('The Hadrons or Path_Wicktract or Path_Perambulator cannot be None')
        if (Path_ModeDoublet is None) and (Path_ModeTriplet is None):
            raise ValueError('At least there must be a path for either Path_ModeTriplet or Path_ModeDoublet')
        self.SinkTime          = SinkTime
        self.SourceTime        = SourceTime
        self.Hadrons           = Hadrons
        self.Path_Wicktract    = Path_Wicktract
        self.Path_Perambulator = Path_Perambulator
        self.Path_ModeDoublet  = Path_ModeDoublet
        self.Path_ModeTriplet  = Path_ModeTriplet
        #To be modified
        self.useGPU            = useGPU
        self.device            = get_best_device(use_gpu = self.useGPU, device_id = Device_ID, verbose = True)

        # Construct now the Perambulator_Super_Tensor
        with h5py.File(self.Path_Perambulator, 'r') as yunus:
            yunus1        = yunus[f'/PerambulatorData/srcTime{self.SourceTime}_snkTime{self.SinkTime}']
            N             = int(np.sqrt( yunus1['srcSpin1']['snkSpin1']['re'].shape[0]))
            P_SuperTensor = torch.zeros((4, 4, N, N), dtype=complex).to(self.device)
            for i in range(4):
                for j in range(4):
                    P_SuperTensor[i, j, :, :] = torch.complex(
                        torch.from_numpy(
                            yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['re'][:]).reshape(N, N), 
                        torch.from_numpy(
                            yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['im'][:]).reshape(N, N)).to(self.device)
            #P^{s_{snk} s_{src} snkevn srcevn}
            self.P_SuperTensor = P_SuperTensor
            print(r'Perambulator_Tensor has been successfully constructed')

        # Construct now the ModeDoublet_Super_Tensor
        if self.Path_ModeDoublet is not None:
            with h5py.File(self.Path_ModeDoublet, 'r') as yunus:
                yunus1         = yunus['/ModeDoubletData']
                MD_SuperTensor = {}
                for group in yunus1:
                    MD_SuperTensor[yunus1[group]] = torch.complex(
                        torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N),
                        torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N)).to(self.device)
                #G^{i j}
            self.MD_SuperTensor = MD_SuperTensor
            print(r'MD_Tensor has been successfully constructed')

        # Construct now the ModeTriplet_Super_Tensor
        if self.Path_ModeTriplet is not None:
            with h5py.File(self.Path_ModeTriplet, 'r') as yunus:
                yunus1         = yunus['/ModeTripletData']
                MT_SuperTensor = {}
                for group in yunus1:
                    MT_SuperTensor[yunus1[group]] = torch.complex(
                        torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N,N),
                        torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N,N)).to(self.device)
                #G^{i j}
            self.MT_SuperTensor = MT_SuperTensor
            print(r'MT_Tensor has been successfully constructed')

        # Cluster the Diagrams
        self.clusters, self.WT_numerical_factors = cluster_extractor(Path_Diagrams = self.Path_Wicktract)

        # SpinStructure Combinations between the hadrons
        self.hadron_product = hadron_info_multiplier(*self.Hadrons)
        print('All combinations of hadron structures coefficients were generated')

        print('Insert now these combinations explicitly into the the clusters!')
        self.clusters_with_kies = [((outer_key, inner_key), 
                                    Final_Perambulator_Container(prpm_container, self.hadron_product).getExplicit_Perambulator_Containers() ) 
                               for outer_key, inner_dict in self.clusters.items() 
                               for inner_key, prpm_container in inner_dict.items()]
        print('Each cluster is now splitted into many clusters with various explicit spin combinations')
# self.clusters_with_kies contains now the following: [((outer_key, inner_key), Topology), ....]
# where Topology is of the form: [PC1, PC2, ...]
# where PCi = [Explicit_Perambulator1, Explicit_Perambulator2,...]
# and in Toplogy after contracting each PCi with the corresponding Tensor all results need to be summed with each others!
# I.e. Topology is actually sum(PCi)
    def getclusters_with_kies(self):
        return self.clusters_with_kies