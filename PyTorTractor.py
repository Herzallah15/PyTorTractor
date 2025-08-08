import numpy as np
import torch
from torch import nn
import h5py
from itertools import product
from functools import reduce
from TorClasses import *
from contractions_handler import *
from Hadrontractions_Converter import *
from Hadron_Info_Converter import *



# GPU Installation needs to be corrected and optimized
def get_best_device(use_gpu=True):
    """
    Liefert automatisch das beste verfügbare Device:
    - 'cuda' für NVIDIA
    - 'mps'  für Apple Silicon
    - 'cpu'  wenn keine GPU oder use_gpu=False
    """
    if not use_gpu:
        return torch.device("cpu")

    # NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon GPU (MPS)
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # Fallback: CPU
    return torch.device("cpu")
# GPU Installation needs to be corrected and optimized


class PyCorrTorch:
    def __init__(self, SinkTime = None, SourceTime = None, 
                 Hadrons = None, Path_Wicktract = None, 
                 Path_Perambulator = None, Path_ModeDoublet = None, Path_ModeTriplet = None, useGPU = False):
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
        device = get_best_device(useGPU)
        #To be modified

        # Construct now the Perambulator_Super_Tensor
        with h5py.File(self.Path_Perambulator, 'r') as yunus:
            yunus1        = yunus[f'/PerambulatorData/srcTime{self.SourceTime}_snkTime{self.SinkTime}']
            N             = int(np.sqrt( yunus1['srcSpin1']['snkSpin1']['re'].shape[0]))
            P_SuperTensor = torch.zeros((4, 4, N, N), dtype=complex).to(device)
            for i in range(4):
                for j in range(4):
                    P_SuperTensor[i, j, :, :] = torch.complex(
                        torch.from_numpy(
                            yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['re'][:]).reshape(N, N), 
                        torch.from_numpy(
                            yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['im'][:]).reshape(N, N)).to(device)
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
                        torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N)).to(device)
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
                        torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N,N)).to(device)
                #G^{i j}
            self.MT_SuperTensor = MT_SuperTensor
            print(r'MT_Tensor has been successfully constructed')

        # Cluster the Diagrams
        self.clusters, self.WT_numerical_factors = cluster_extractor(Path_Diagrams = self.Path_Wicktract)

        # SpinStructure Combinations between the hadrons
        self.hadron_product = hadron_info_multiplier(*self.Hadrons)
        print('All combinations of hadron structures coefficients were generated')
        print('Insert now these combinations explicitly into the the clusters!')
        clusters_with_paths = [((outer_key, inner_key), prpm_container) 
                               for outer_key, inner_dict in self.clusters.items() 
                               for inner_key, prpm_container in inner_dict.items()]
        
        
