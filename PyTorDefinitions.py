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

erp_0 = 'Unrecognized path. When including displacement, the path is expected to be of the form px(int)_py(int)_pz(int)_ddir(int)_dlen(int)_t(int)'
erp_1 = 'When including displacement, the path is expected to be of the form px(int)_py(int)_pz(int)_ddir0_dlen0_t(int)'
erp_2 = 'Unrecognized path. The path is expected to be px(int)_py(int)_pz(int)_ddir0_t(int)'
erp_3 = 'For unspecified dlen or dlen = 0, the displacement is expected to be zero/ For ddir0 dlen is expected to be dlen0 or None'
erp_4 = 'For non-zero displacement, ddlen is expected to be non-zero'


# path = "pxI_pyJ_pzK_ddirT_(dlenM_)tU"
def check_path_MDT(path):
    path_sp = path.split('_')
    #path_sp = ['pxI', 'pyJ', 'pzK', 'ddirT', ('dlenM'), 'tU']
    if len(path_sp) == 6:
        correct_path = path_sp[0][:2]+path_sp[1][:2]+path_sp[2][:2]+path_sp[3][:4]+path_sp[4][:4]+path_sp[5][0]
        if correct_path != 'pxpypzddirdlent':
            raise ValueError(path, '. ', erp_0)
        if (path_sp[3][4:] == '0') and (not path_sp[4][4:] == '0'):
            raise ValueError(erp_3)
        if (path_sp[3][4:] != '0') and (path_sp[4][4:] == '0'):
            raise ValueError(path, '. ', erp_4)
    elif len(path_sp) == 5:
        correct_path = path_sp[0][:2]+path_sp[1][:2]+path_sp[2][:2]+path_sp[3][:4]+path_sp[4][0]
        if correct_path != 'pxpypzddirt':
            raise ValueError(path, '. ', erp_2)
        if path_sp[3][4:] != '0':
            raise ValueError(path, '. ', erp_3)
    else:
        raise ValueError('Unrecognized path')
#def momentum(string_value):
#    if string_value == 'mom_ray_000':
#        return 'px0_py0_pz0'

#            "0" =>  0
#            "+" =>  1
#            "-" => -1
#            "#" =>  2
#            "=" => -2
#            "T" =>  3
#            "t" => -3
mom_map = {0: '0', 1: '+', -1: '-', 2: '#', -2: '=', 3: 'T', -3: 't'}
def sgn(x):
    if x > 0:
        return '+'
    elif x<0:
        return '-'
    else:
        return '0'
def momentum(p_momentum):
    p = list(p_momentum)
    px, py, pz = p
    pab = [np.abs(pi) for pi in p]
    pxa, pya, pza = pab
    p_value = f'px{px}_py{py}_pz{pz}'
    string_value = 'mom_ray_'
    if all(pi == 0 for pi in p):
        return {'mom_path': 'mom_ray_000', 'int_value': p_value}
    if len(set(pab)) == 1:#three terms are equal
        string_value += f'{sgn(px)}{sgn(py)}{sgn(pz)}'
    elif len(set(pab)) == 2:#two terms are equal
        if pxa == pya:
            string_value += f'{sgn(px)}{sgn(py)}{mom_map[pz]}'
        elif pxa == pza:
            string_value += f'{sgn(px)}{mom_map[py]}{sgn(pz)}'
        elif pya == pza:
            string_value += f'{mom_map[px]}{sgn(py)}{sgn(pz)}'    
    else:#none of the terms are equal!
        string_value += f'{mom_map[px]}{mom_map[py]}{mom_map[pz]}'
    return {'mom_path': string_value, 'int_value': p_value}

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

    (1,3,0): 'j',
    (1,3,1): 'k',

    (0,0,0): 'm',
    (0,0,1): 'n',
    (0,0,2): 'o',

    (0,1,0): 'p',
    (0,1,1): 'q',
    (0,1,2): 'r',

    (0,2,0): 's',
    (0,2,1): 't',
    (0,2,2): 'u',

    (0,3,0): 'v',
    (0,3,1): 'w'
}


spin_index_map = {
    (1,0,0): 'A',
    (1,0,1): 'B',
    (1,0,2): 'C',

    (1,1,0): 'D',
    (1,1,1): 'E',
    (1,1,2): 'F',

    (1,2,0): 'G',
    (1,2,1): 'H',
    (1,2,2): 'I',

    (1,3,0): 'J',
    (1,3,1): 'K',

    (0,0,0): 'M',
    (0,0,1): 'N',
    (0,0,2): 'O',

    (0,1,0): 'P',
    (0,1,1): 'Q',
    (0,1,2): 'R',

    (0,2,0): 'S',
    (0,2,1): 'T',
    (0,2,2): 'U',

    (0,3,0): 'V',
    (0,3,1): 'W'
}
MDT_index_map = {
    (1,0): 'x',
    (1,1): 'y',
    (1,2): 'z',
    (1,3): 'l',
    (0,0): 'X',
    (0,1): 'Y',
    (0,2): 'Z',
    (0,3): 'L'
}



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

'''
def Tensor_Product(Qs):
    reshaped = [
        q.view(*([1] * (2 * i)), q.shape[0], q.shape[1], *([1] * (2 * (len(Qs) - i - 1))))
        for i, q in enumerate(Qs)
    ]
    return reduce(operator.mul, reshaped)
'''

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


def PyTor_MDoublet(Path_ModeDoublet = None, Device = None, cplx128 = True, Selected_Groups = None):
    if cplx128:
        data_type = torch.complex128
    else:
        data_type = torch.complex64
    MD_SuperTensor_Dict = {}
    with h5py.File(Path_ModeDoublet, 'r') as yunus:
        yunus1 = yunus['/ModeDoubletData']
        if Selected_Groups is not None:
            if isinstance(Selected_Groups, str):
                selected_pathes = [Selected_Groups]
            else:
                selected_pathes = Selected_Groups
        else:
            selected_pathes = [group for group in yunus1]
        if any(group.split('_')[4].startswith("dlen") for group in selected_pathes):
            test_the_path = [check_path_MDT(test_path) for test_path in selected_pathes]
            del test_the_path
        for i, group in enumerate(selected_pathes):
            if i == 0:
                if yunus1[group]['re'].ndim == 1:
                    N    = int(np.sqrt(yunus1[group]['re'][:].shape[0]))
                    resp = True
                elif yunus1[group]['re'].ndim == 2:
                    N    = yunus1[group]['re'][:].shape[0]
                    resp = False
                else:
                    raise ValueError('Failed to read the ModeDoublets')
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

def PyTor_MTriplet(Path_ModeTriplet = None, Device = None, cplx128 = True, Selected_Groups = None, Use_Triplet_Identity = None):
    if cplx128:
        data_type = torch.complex128
    else:
        data_type = torch.complex64
    if (Use_Triplet_Identity is not None) and not isinstance(bool, str, set, list, tuple):
        raise TypeError(f'The argument Use_Triplet_Identity can be either bool, str, set, list or tuple!')
    MT_SuperTensor_Dict = {}
    with h5py.File(Path_ModeTriplet, 'r') as yunus:
        yunus1 = yunus['/ModeTripletData']
        if Selected_Groups is not None:
            if isinstance(Selected_Groups, str):
                selected_pathes = [Selected_Groups]
            else:
                selected_pathes = Selected_Groups
        else:
            selected_pathes = [group for group in yunus1]
        if any(group.split('_')[4].startswith("dlen") for group in selected_pathes):
            test_the_path = [check_path_MDT(test_path) for test_path in selected_pathes]
            del test_the_path
        for i, group in enumerate(selected_pathes):
            if i == 0:
                if yunus1[group]['re'].ndim == 1:
                    N    = int(np.cbrt(yunus1[group]['re'][:].shape[0]))
                    resp = True
                elif yunus1[group]['re'].ndim == 3:
                    N    = yunus1[group]['re'][:].shape[0]
                    resp = False
                else:
                    raise ValueError('Failed to read the ModeDoublets')
            if resp:
                MT_SuperTensor_Dict[group] = torch.complex(
                    torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N,N),
                    torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N,N)).to(dtype=data_type).to(Device)
            else:
                MT_SuperTensor_Dict[group] = torch.complex(
                    torch.from_numpy(yunus1[group]['re'][:]),
                    torch.from_numpy(yunus1[group]['im'][:])).to(dtype=data_type).to(Device)
    if Use_Triplet_Identity is not None:
        if isinstance(Use_Triplet_Identity, bool):
            if Use_Triplet_Identity:
                re_construct_group = [group for group in MT_SuperTensor_Dict]
            else:
                print(r'MT_Tensor has been successfully constructed')
                return MT_SuperTensor_Dict
        elif isinstance(Use_Triplet_Identity, str):
            re_construct_group = [Use_Triplet_Identity]
        elif isinstance(Use_Triplet_Identity, (set, list, tuple)):
            re_construct_group = list(Use_Triplet_Identity)
        for group in re_construct_group:
            if group.split('_')[3].split('ddir')[1] != '0':
                displacement_q2 = group.split('ddir')[0]+'ddir0_'+group.split('_')[3].split('ddir')[1]+'_0_'+group.split('_')[-1]
                displacement_q3 = group.split('ddir')[0]+'ddir0_0_'+group.split('_')[3].split('ddir')[1]+'_'+group.split('_')[-1]
                MT_SuperTensor_Dict[displacement_q2] = -1 * MT_SuperTensor_Dict[group].permute(1,0,2)
                MT_SuperTensor_Dict[displacement_q3] = MT_SuperTensor_Dict[group].permute(2,0,1)
                print(f'The groups {displacement_q2} and {displacement_q3} have been constructed from {group}')
    print(r'MT_Tensor has been successfully constructed')
    return MT_SuperTensor_Dict

def Prmp_Set(exp_prmp_container, Stack_List = None):
    Spin_Indices = []
    seen_hadron  = set()
    num_factor   = 1.0
    for perambulator in exp_prmp_container:
        s,s_Bar = perambulator.getS() - 1, perambulator.getS_Bar() - 1
        Spin_Indices.append(s)
        Spin_Indices.append(s_Bar)
        if perambulator.getH() not in seen_hadron:
            seen_hadron.add(perambulator.getH())
            num_factor *= perambulator.getFF_H()
        if perambulator.getH_Bar() not in seen_hadron:
            seen_hadron.add(perambulator.getH_Bar())
            num_factor *= perambulator.getFF_H_Bar()
    if Stack_List is not None:
        vor_list = []
        for stacked_index in Stack_List:
            if stacked_index is not None:
                vor_list.appned(stacked_index)
        if len(vor_list) == 0:
            raise ValueError('Something wrong with the stacked Modes')
        Spin_Indices = vor_list + Spin_Indices
    return {'Spin_Indices': Spin_Indices, 'Numerical_Factor': num_factor}

           
def SpnFF_XTractor(full_cluster):
    #Full cluster is a list. Each element is a perambulator_container.
    #From each element of this list we extract: 1. Explicit-Spins. 2. Overall number
    return [Prmp_Set(exp_prmp_container) for exp_prmp_container in full_cluster]
    
def SpnFF_SXTractor(full_cluster, Stack_Lists):
    #Full cluster is a list. Each element is a perambulator_container.
    #From each element of this list we extract: 1. Explicit-Spins. 2. Overall number
    #Stack_Lists is a list of lists of the stacked indices!
    n = len(full_cluster)
    m = len(Stack_Lists)
    if n != m:
        raise ValueError('Something wrong with the stacked Modes. Unequal number of combinations..')
    return [Prmp_Set(full_cluster[i], Stack_List = Stack_Lists[i]) for i in range(n)]    
    
    
def co_to_Hadorn_co(list_of_Hadrons, Full_Map_Of_Hadrons):
    #list_of_hadrons is of the form ((1,0), ...)
    all_hadrons = []
    for one_hadron in list_of_Hadrons:
        all_hadrons.append(Full_Map_Of_Hadrons[one_hadron])
    return all_hadrons


def pick_combis(hadron_cluster, all_combinations_map):
    #hadron_clzster is of the form ((1,0),...)
    #all_combinations_map is a map of the form: {((1,0), ..): [combi_i], ..}
    found = 0
    for one_cluster in all_combinations_map:
        if set(hadron_cluster) == set(one_cluster):
            found += 1
            one_cluster_combi = all_combinations_map[one_cluster]
    if found == 1:
        return one_cluster_combi
    else:
        raise ValueError('Failed to extract the cluster')


def Perambulator_Laph(all_perambulators, exp_prmp_container, snktime, srctime):
    Prmp_Indices_In  = ''
    Prmp_Indices_Out = ''
    Prmp_Tensors = []
    for perambulator in exp_prmp_container:
        Q_Info, Q_Bar_Info = perambulator.getQ(), perambulator.getQ_Bar()
        Prmp_Indices_In   += spin_index_map[Q_Info] + spin_index_map[Q_Bar_Info]
        Prmp_Indices_In   += index_map[Q_Info] + index_map[Q_Bar_Info] + ','
        Prmp_Indices_Out  += spin_index_map[Q_Info] + spin_index_map[Q_Bar_Info]
        p_left, p_right   = perambulator.getH()[0], perambulator.getH_Bar()[0]
        prmp_flavor       = perambulator.getFlavor()
        if p_left   == 1 and p_right == 0:
            time    = f'srcTime{srctime}_snkTime{snktime}'
        elif p_left == 0 and p_right == 1:
            time    = f'srcTime{snktime}_snkTime{srctime}'
        elif p_left == 1 and p_right == 1:
            time    = f'srcTime{snktime}_snkTime{snktime}'
        elif p_left == 0 and p_right == 0:
            time    = f'srcTime{srctime}_snkTime{srctime}'
        else:
            raise ValueError('Error in extracting perambulators from the Perambulator_Tensor_Dict')
        Prmp_Tensors.append(all_perambulators[prmp_flavor][time])
    return {'index_In': Prmp_Indices_In[:-1], 'index_Out': Prmp_Indices_Out, 'Tensor': Prmp_Tensors}

def MDT_Laph(MDT_Info = None, snktime = None, srctime=None, ModeD = None, ModeT = None):
    M_Tensors = []
    for path in MDT_Info:
        if path[0] == '0':
            final_path = path[3:]+'_t'+str(srctime)
            if path[1] == 'D':
                M_Tensors.append(ModeD[final_path].conj())
            elif path[1] == 'T':
                M_Tensors.append(ModeT[final_path].conj())
            else:
                raise ValueError('Failed to identiy type of the Mode')
        elif path[0] == '1':
            final_path = path[3:]+'_t'+str(snktime)
            if path[1] == 'D':
                M_Tensors.append(ModeD[final_path])
            elif path[1] == 'T':
                M_Tensors.append(ModeT[final_path])
            else:
                raise ValueError('Failed to identiy type of the Mode')
        else:
            raise ValueError('Failed to identify sink and source times')
    return M_Tensors

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

