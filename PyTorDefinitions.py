import numpy as np
import torch
from typing import Optional
from torch import nn
import platform
import warnings
import h5py
from itertools import product
from functools import reduce
import operator
import copy
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

    (1,3,0): 'G',
    (1,3,1): 'H',
    (1,3,2): 'I',

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
    (2,1,2): 'x',

    (2,2,0): 'V',
    (2,2,1): 'W',
    (2,2,2): 'X',

    (3,0,0): 'y',
    (3,0,1): 'z',
    (3,0,2): 'Y',

    (3,1,0): 'S',
    (3,1,1): 'T',
    (3,1,2): 'U',

}


def ddir(path):
    if np.all(path == np.array([0,0,0])):
        return 'ddir0'
    else:
        raise ValueError('Displacement_CannotBe')


def momentum(string_value):
    if string_value == 'mom_ray_000':
        return 'px0_py0_pz0'


def hdrn_type(x):
    if x == 'meson_operators':
        return 'M'
    elif x == 'baryon_operators':
        return 'B'

def srcsnk_exchanger(group):
    parts = group.split('_')
    srctime = parts[0].replace('srcTime', '')
    snktime = parts[1].replace('snkTime', '')
    return f'srcTime{snktime}_snkTime{srctime}'

def get_best_device(use_gpu: bool = True, device_id: Optional[int] = None, verbose: bool = True, cplx128 = True) -> torch.device:
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
            if cplx128:
                if verbose:
                    print("Apple MPS backend detected, but CPU is in use (due to complex128 support).")
                return torch.device("cpu")
            else:
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


def Tensor_Product(Qs):
    reshaped = [
        q.view(*([1] * (2 * i)), q.shape[0], q.shape[1], *([1] * (2 * (len(Qs) - i - 1))))
        for i, q in enumerate(Qs)
    ]
    return reduce(operator.mul, reshaped)


def gamma(i, datatype):
    if i == 5:
        gamma5 = torch.tensor([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=datatype)
        return gamma5
    elif i == 4:
        gamma4 = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ], dtype=datatype)
        return gamma4


def combine_all(all_info):
    topologies       = all_info[0]
    numerical_factor = all_info[1]
    final_result     = {}
    for topology in topologies:
        for dgrm_nmbr in topology[0][1]:
            if dgrm_nmbr not in final_result:
                final_result[dgrm_nmbr] = []
            final_result[dgrm_nmbr].append(topology[1])
    res = 0
    for dgrm_nmbr in final_result:
        product = 1
        for prdkt in final_result[dgrm_nmbr]:
            product *= prdkt.item()
        res += (numerical_factor[dgrm_nmbr].item()) * product
    return res




error01 = 'The Hadrons or Path_Wicktract cannot be None'
error1  = 'Perambulators cannot be None. They must be of the form {Light: Perambulator_dict, Strange: Perambulator_dict, Charm: Perambulator_dict}'
error02 = 'At least there must be either ModeTriplets or ModeDoublets'


def PyTor_Perambulator(Path_Perambulator = None, Device = None, Double_Reading = False, cplx128 = True):
    if cplx128:
        data_type = torch.complex128
    else:
        data_type = torch.complex64
    P_SuperTenspr_Dict = {}
    seen_gropus        = set()
    if isinstance(Path_Perambulator, str):
        Path_All_Perambulators = [Path_Perambulator]
    else:
        Path_All_Perambulators = Path_Perambulator
    for One_Path_Perambulator in Path_All_Perambulators:
        with h5py.File(One_Path_Perambulator, 'r') as yunus:
            yunus01 = yunus[f'/PerambulatorData']
            for t_c, srcTime_snkTime_group in enumerate(yunus01):
                if srcTime_snkTime_group in seen_gropus and not Double_Reading:
                    print(f'The group {srcTime_snkTime_group} appears more than one time! Add Double_Reading = True')
                    print('or provide Path_Perambulator that contain only unique groups of srcTime_snkTime')
                    raise ValueError('Error in reading data from Path_Perambulator')
                else:
                    seen_gropus.add(srcTime_snkTime_group)
                yunus1        = yunus01[f'{srcTime_snkTime_group}']
                if t_c == 0:
                    if yunus1['srcSpin1']['snkSpin1']['re'].ndim == 1:
                        N             = int(np.sqrt(yunus1['srcSpin1']['snkSpin1']['re'].shape[0]))
                        resp          = True
                    elif yunus1['srcSpin1']['snkSpin1']['re'].ndim == 2:
                        N             = yunus1['srcSpin1']['snkSpin1']['re'].shape[0]
                        resp          = False
                    else:
                        raise ValueError('Failed to read Perambulators!')
                P_SuperTensor = torch.zeros((4, 4, N, N), dtype=data_type)
                for i in range(4):
                    for j in range(4):
                        if resp:
                            P_SuperTensor[i, j, :, :] = torch.complex(
                                torch.from_numpy(
                                    yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['re'][:]).reshape(N, N), 
                                torch.from_numpy(
                                    yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['im'][:]).reshape(N, N))
                        else:
                            P_SuperTensor[i, j, :, :] = torch.complex(
                                torch.from_numpy(
                                    yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['re'][:]), 
                                torch.from_numpy(
                                    yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['im'][:]))     
                P_SuperTenspr_Dict[srcTime_snkTime_group] = P_SuperTensor.to(Device)
            for srcTime_snkTime_group in seen_gropus:
                ex_srcsnk = srcsnk_exchanger(srcTime_snkTime_group)
                if ex_srcsnk not in seen_gropus:
                    g5                            = gamma(5, data_type).to(Device)
                    g4                            = gamma(4, data_type).to(Device)
                    gM                            = torch.matmul(g5, g4)
                    P_SuperTenspr_Dict[ex_srcsnk] = torch.einsum('ij,jnlm,nk->kiml', gM, P_SuperTenspr_Dict[srcTime_snkTime_group], gM).conj()
                    print(f' Perambulator for {ex_srcsnk} has been constructed using infos about {srcTime_snkTime_group}!')
    print(r'Perambulator_Tensor has been successfully constructed')
    return P_SuperTenspr_Dict


def PyTor_MDoublet(Path_ModeDoublet = None, Device = None, Double_Reading = False, cplx128 = True):
    if cplx128:
        data_type = torch.complex128
    else:
        data_type = torch.complex64
    MD_SuperTensor_Dict = {}
    seen_gropus         = set()
    if isinstance(Path_ModeDoublet, str):
        Path_All_ModeDoublet = [Path_ModeDoublet]
    else:
        Path_All_ModeDoublet = Path_ModeDoublet
    for One_Path_ModeDoublet in Path_All_ModeDoublet:
        with h5py.File(One_Path_ModeDoublet, 'r') as yunus:
            yunus1         = yunus['/ModeDoubletData']
            for i, group in enumerate(yunus1):
                if i == 0:
                    if yunus1[group]['re'].ndim == 1:
                        N    = int(np.sqrt(yunus1[group]['re'][:].shape[0]))
                        resp = True
                    elif yunus1[group]['re'].ndim == 2:
                        N    = yunus1[group]['re'][:].shape[0]
                        resp = False
                    else:
                        raise ValueError('Failed to read the ModeDoublets')
                if group in seen_gropus and not Double_Reading:
                    print(f'The group {group} appears more than one time! Add Double_Reading = True')
                    print('or provide Path_ModeDoublet that contain only unique groups of groups')
                    raise ValueError('Error in reading data from Path_ModeDoublet')
                if resp:
                    MD_SuperTensor_Dict[group] = torch.complex(
                        torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N),
                        torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N)).to(dtype=data_type).to(Device)
                else:
                    MD_SuperTensor_Dict[group] = torch.complex(
                        torch.from_numpy(yunus1[group]['re'][:]),
                        torch.from_numpy(yunus1[group]['im'][:])).to(dtype=data_type).to(Device)              
    print(r'MD_Tensor has been successfully constructed')
    return MD_SuperTensor_Dict

def PyTor_MTriplet(Path_ModeTriplet = None, Device = None, Double_Reading = False, cplx128 = True):
    if cplx128:
        data_type = torch.complex128
    else:
        data_type = torch.complex64
    MT_SuperTensor_Dict = {}
    seen_gropus         = set()
    if isinstance(Path_ModeTriplet, str):
        Path_All_ModeTriplet = [Path_ModeTriplet]
    else:
        Path_All_ModeTriplet = Path_ModeTriplet
    for One_Path_ModeTriplet in Path_All_ModeTriplet:
        with h5py.File(One_Path_ModeTriplet, 'r') as yunus:
            yunus1         = yunus['/ModeTripletData']
            for i, group in enumerate(yunus1):
                if i == 0:
                    if yunus1[group]['re'].ndim == 1:
                        N    = int(np.cbrt(yunus1[group]['re'][:].shape[0]))
                        resp = True
                    elif yunus1[group]['re'].ndim == 3:
                        N    = yunus1[group]['re'][:].shape[0]
                        resp = False
                    else:
                        raise ValueError('Failed to read the ModeDoublets')
                if group in seen_gropus and not Double_Reading:
                    print(f'The group {group} appears more than one time! Add Double_Reading = True')
                    print('or provide Path_ModeTriplet that contain only unique groups of groups')
                    raise ValueError('Error in reading data from Path_ModeTriplet')
                if resp:
                    MT_SuperTensor_Dict[group] = torch.complex(
                        torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N,N),
                        torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N,N)).to(dtype=data_type).to(Device)
                else:
                    MT_SuperTensor_Dict[group] = torch.complex(
                        torch.from_numpy(yunus1[group]['re'][:]),
                        torch.from_numpy(yunus1[group]['im'][:])).to(dtype=data_type).to(Device)     
    print(r'MT_Tensor has been successfully constructed')
    return MT_SuperTensor_Dict








#comment_01:
# self.clusters_with_kies contains now the following: [((outer_key, inner_key), Topology), ....]
# where Topology is of the form: [PC1, PC2, ...]
# where PCi = [Explicit_Perambulator1, Explicit_Perambulator2,...]
# and in Toplogy after contracting each PCi with the corresponding Tensor all results need to be summed with each others!
# I.e. Topology is actually sum(PCi)
# outer cluster is somethong of the form: ((0, 1), (0, 0), (1, 0), (1, 1))
# exp_prmp_container is of the form [ExplicitPerambulator, ExplicitPerambulator, ...]
# Perambulators is of the form {'Light': Perambulator_dict, 'Strange': Perambulator_dict, 'Charm': Perambulator_dict}

#comment_02:
# exp_prmp_container is of the form [ExplicitPerambulator, ExplicitPerambulator, ...]

#PART_OUTCOMMENT_0
        #self.Path_Wicktract    = Path_Wicktract
        #self.Path_Perambulator = Path_Perambulator
        #self.Path_ModeDoublet  = Path_ModeDoublet
        #self.Path_ModeTriplet  = Path_ModeTriplet


        #self.useGPU            = useGPU
        #self.device            = get_best_device(use_gpu = self.useGPU, device_id = Device_ID, verbose = True)
        # Construct now the Perambulator_Super_Tensor
        #with h5py.File(self.Path_Perambulator, 'r') as yunus:
        #    yunus1        = yunus[f'/PerambulatorData/srcTime{self.SourceTime}_snkTime{self.SinkTime}']
        #    N             = int(np.sqrt(yunus1['srcSpin1']['snkSpin1']['re'].shape[0]))
        #    P_SuperTensor = torch.zeros((4, 4, N, N), dtype=torch.complex128)
        #    for i in range(4):
        #        for j in range(4):
        #            P_SuperTensor[i, j, :, :] = torch.complex(
        #                torch.from_numpy(
        #                    yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['re'][:]).reshape(N, N), 
        #                torch.from_numpy(
        #                    yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['im'][:]).reshape(N, N))
        #    #P^{s_{snk} s_{src} snkevn srcevn}
        #    g5                    = gamma(5, torch.complex128).to(self.device)
        #    g4                    = gamma(4, torch.complex128).to(self.device)
        #    gM                    = torch.matmul(g5, g4)
        #    self.P_SuperTensor    = P_SuperTensor.to(self.device)
        #    self.P_Re_SuperTensor = torch.einsum('ij,jnlm,nk->kiml', gM, self.P_SuperTensor, gM).conj()
        #    print(r'Perambulator_Tensor has been successfully constructed')

        #   # Construct now the ModeDoublet_Super_Tensor
        #if self.Path_ModeDoublet is not None:
        #    with h5py.File(self.Path_ModeDoublet, 'r') as yunus:
        #        yunus1         = yunus['/ModeDoubletData']
        #        MD_SuperTensor = {}
        #        for group in yunus1:
        #            MD_SuperTensor[group] = torch.complex(
        #                torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N),
        #                torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N)).to(dtype=torch.complex128).to(self.device)
        #        #G^{i j}
        #    self.MD_SuperTensor = MD_SuperTensor
        #    print(r'MD_Tensor has been successfully constructed')

        #   # Construct now the ModeTriplet_Super_Tensor
        #if self.Path_ModeTriplet is not None:
        #    with h5py.File(self.Path_ModeTriplet, 'r') as yunus:
        #        yunus1         = yunus['/ModeTripletData']
        #        MT_SuperTensor = {}
        #        for group in yunus1:
        #            MT_SuperTensor[group] = torch.complex(
        #                torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N,N),
        #                torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N,N)).to(dtype=torch.complex128).to(self.device)
        #        #G^{i j}
        #    self.MT_SuperTensor = MT_SuperTensor
        #    print(r'MT_Tensor has been successfully constructed')

