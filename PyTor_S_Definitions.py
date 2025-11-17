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
from PyTorDefinitions import *


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
    (1,3,2): 'l',

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
    (0,3,1): 'w',
    (0,3,2): 'x'
}
stack_index_map = {0: 'y',
                   1: 'z',
                   2: 'A',
                   3: 'B',
                   4: 'C',
                   5: 'D',
                   6: 'E',
                   7: 'F',
                   8: 'G',
                   9: 'H',
                   10: 'I',
                   11: 'J',
                   12: 'K',
                   13: 'L',
                   14: 'N',
                   15: 'O',
                   16: 'B',
                   17: 'Q',
                   18: 'R'
                  }
SG_Cindex_map = {
    (2,0,0): 'Z',
    (2,0,1): 'Z',
    
    (2,1,0): 'X',
    (2,1,1): 'X',
    
    (2,2,0): 'V',
    (2,2,1): 'V',
    
    (2,3,0): 'T',
    (2,3,1): 'T'
}
SG_Xindex_map = {    
    (2,0,0): 'Y',
    (2,0,1): 'Y',
    
    (2,1,0): 'W',
    (2,1,1): 'W',
    
    (2,2,0): 'U',
    (2,2,1): 'U',
    
    (2,3,0): 'S',
    (2,3,1): 'S'
}
def stacker(P_Naive):
    return [torch.stack(list(row)) for row in zip(*P_Naive)]
def TorchPhase(MomentumTuple, CoordinateTuple0, Coordinates_Meta, Device, DType, Ngrid):
    CoordinateTuple1 = Coordinates_Meta['Coordinates']
    P_Meta = Coordinates_Meta['Ls']
    MomentumTuple = MomentumTuple[0]
    MomentumTuple = [2 * torch.pi * (MomentumTuple[i] / P_Meta[i]) for i in range(3)]
    grid_coordinates = []
    if Ngrid != len(CoordinateTuple1):
        raise ValueError('Failed to identiy the grid coordinates')
    for i in range(Ngrid):
        SP3 = 0j
        for k in range(3):
            SP3 += MomentumTuple[k] * (CoordinateTuple0[k] + CoordinateTuple1[i][k])
        grid_coordinates.append(SP3)
    grid_tensor = torch.tensor(grid_coordinates, device=Device, dtype=DType)
    return torch.exp(-1j * grid_tensor)
def Perambulator_Extractor(all_perambulators = None, all_SG_perambulators = None, 
                           exp_prmp_container = None, snktime = None, srctime = None, current_time=None, Hadron_Momenta = None):
    Prmp_Indices_In  = ''
    Prmp_Tensors = []
    seen_hadron  = set()
    spars_perambulators = []
    for perambulator in exp_prmp_container:
        if (perambulator.getH()[0] in [2, 3]) or (perambulator.getH_Bar()[0] in [2, 3]):
            spars_perambulators.append(perambulator)
            continue
        num_factor   = 1.0
        if perambulator.getH() not in seen_hadron:
            seen_hadron.add(perambulator.getH())
            num_factor *= perambulator.getFF_H()
        if perambulator.getH_Bar() not in seen_hadron:
            seen_hadron.add(perambulator.getH_Bar())
            num_factor *= perambulator.getFF_H_Bar()
        s,s_Bar = perambulator.getS() - 1, perambulator.getS_Bar() - 1
        Q_Info, Q_Bar_Info = perambulator.getQ(), perambulator.getQ_Bar()
        Prmp_Indices_In   += index_map[Q_Info] + index_map[Q_Bar_Info] + ','
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
        Prmp_Tensors.append(all_perambulators[prmp_flavor][time][s, s_Bar, :, :] * num_factor)
    if all_SG_perambulators is not None:
        Geomertry = {'Light': {}, 'Strange': {}, 'Charm': {}}
        for flavors in all_SG_perambulators:
            if all_SG_perambulators[flavors] is not None:
                Geomertry[flavors]['Coordinate_Offset'] = all_SG_perambulators[flavors][1]['Momentum_offsets']
                Geomertry[flavors]['Coordinates'] = all_SG_perambulators[flavors][1]['Momentum_onshell']
                Geomertry[flavors]['Ngrid'] = all_SG_perambulators[flavors][1]['Ngrid']
    sparsgrids_contracter = {}
    for perambulator in spars_perambulators:
        prmp_flavor       = perambulator.getFlavor()
        num_factor   = 1.0
        if perambulator.getH() not in seen_hadron:
            seen_hadron.add(perambulator.getH())
            num_factor *= perambulator.getFF_H()
        if perambulator.getH_Bar() not in seen_hadron:
            seen_hadron.add(perambulator.getH_Bar())
            num_factor *= perambulator.getFF_H_Bar()
        s,s_Bar = perambulator.getS() - 1, perambulator.getS_Bar() - 1
        Q_Info, Q_Bar_Info = perambulator.getQ(), perambulator.getQ_Bar()
        if (perambulator.getH()[0] in [2, 3]) and (perambulator.getH_Bar()[0] in [0, 1]):
            htime = srctime if perambulator.getH_Bar()[0] == 0 else snktime
            time         = f'srcTime{htime}_snkTime{current_time}'
            XCI_Indices  = SG_Xindex_map[Q_Info] +  SG_Cindex_map[Q_Info] + index_map[Q_Bar_Info]
            I_Indices    = index_map[Q_Bar_Info]
            Spars_Hadron = tuple(perambulator.getH())
            P_Mom        = tuple(Hadron_Momenta[perambulator.getH()])
        elif (perambulator.getH()[0] in [0, 1]) and (perambulator.getH_Bar()[0] in [2, 3]):
            htime = srctime if perambulator.getH()[0] == 0 else snktime
            time = f'ex_srcTime{htime}_snkTime{current_time}'
            XCI_Indices  = SG_Xindex_map[Q_Bar_Info] +  SG_Cindex_map[Q_Bar_Info] + index_map[Q_Info]
            I_Indices    = index_map[Q_Info]
            Spars_Hadron = tuple(perambulator.getH_Bar())
            P_Mom = tuple(Hadron_Momenta[perambulator.getH_Bar()])
        elif (perambulator.getH()[0] == perambulator.getH_Bar()[0]) and (perambulator.getH()[0] in [2, 3]):
            raise TypeError('Current Version does not handle loop in a current')
        else:
            raise ValueError('Failed to extract SparsGridPerambulator')
        if Spars_Hadron not in sparsgrids_contracter:
            sparsgrids_contracter[Spars_Hadron] = {'SparsP': [], 'in': [], 'out': [], 'Momentum': set(), 'flavor': prmp_flavor}
        sparsgrids_contracter[Spars_Hadron]['SparsP'].append(all_SG_perambulators[prmp_flavor][0][time][s, s_Bar, :, :, :] * num_factor)
        sparsgrids_contracter[Spars_Hadron]['in'].append(XCI_Indices)
        sparsgrids_contracter[Spars_Hadron]['out'].append(I_Indices)
        sparsgrids_contracter[Spars_Hadron]['Momentum'].add(P_Mom)
    if len(sparsgrids_contracter) != 0:
        for SG_P in sparsgrids_contracter:
            Pmom = list(sparsgrids_contracter[SG_P]['Momentum'])
            if len(Pmom) != 1:
                raise ValueError('Failed to access SP_Perambulator Infos')
            if {len(sparsgrids_contracter[SG_P]['SparsP']), len(sparsgrids_contracter[SG_P]['in']), len(sparsgrids_contracter[SG_P]['out'])} != {2}:
                raise ValueError('Failed to access SP_Perambulator Infos')
            sprsI = sparsgrids_contracter[SG_P]['out'][0] + sparsgrids_contracter[SG_P]['out'][1]
            Prmp_Indices_In += sprsI + ','
            sprcCX1 = sparsgrids_contracter[SG_P]['in'][0]
            sprcCX2 = sparsgrids_contracter[SG_P]['in'][1]
            sprsCX  = sprcCX1 + ',' + sprcCX2
            if sprcCX1[:2] != sprcCX2[:2]:
                raise ValueError('Failed to multiply with Momentum Phase')
            Pix = sprcCX1[0]
            sparsDevice, sparsDtype = sparsgrids_contracter[SG_P]['SparsP'][0].device, sparsgrids_contracter[SG_P]['SparsP'][0].dtype
            Coordinate_Offset = Geomertry[sparsgrids_contracter[SG_P]['flavor']]['Coordinate_Offset'][current_time]
            Coordinates = Geomertry[sparsgrids_contracter[SG_P]['flavor']]['Coordinates']
            Ngrid = Geomertry[sparsgrids_contracter[SG_P]['flavor']]['Ngrid']
            if Pmom[0] != (0,0,0):
                Prmp_Tensors.append(torch.einsum(f'{sprsCX},{Pix}->{sprsI}', sparsgrids_contracter[SG_P]['SparsP'][0],
                                                 sparsgrids_contracter[SG_P]['SparsP'][1], TorchPhase(Pmom, Coordinate_Offset, Coordinates,
                                                                                                      sparsDevice, sparsDtype, Ngrid)))
            else:
                Prmp_Tensors.append(torch.einsum(f'{sprsCX}->{sprsI}', sparsgrids_contracter[SG_P]['SparsP'][0],
                                                 sparsgrids_contracter[SG_P]['SparsP'][1]))
    return Prmp_Indices_In[:-1], Prmp_Tensors

def Perambulator_Mode_Handler_PStacked(Full_Cluster = None, All_Mode_Info = None,
                             snktime = None, srctime=None,
                             Prmbltr = None, ModeD = None, ModeT = None):
    # Find all Modes, which are equivalent
    unique_mode_paths = {}
    for i, path in enumerate(All_Mode_Info):
        if tuple(path) in unique_mode_paths:
            unique_mode_paths[tuple(path)]['ExplicitPerambulators'].append(i)
        else:
            unique_mode_paths[tuple(path)] = {'ExplicitModes': [], 'ExplicitPerambulators': []}
            unique_mode_paths[tuple(path)]['ExplicitPerambulators'] = [i]
    unique_mode_paths_copy = [path for path in unique_mode_paths]
    # Now put in the explicit expressions of the modes!
    for unique_path in unique_mode_paths_copy:
        # path here corresponds to one_mode_path
        for path in unique_path:
            if path[0] == '0':
                final_path = path[3:]+'_t'+str(srctime)
                if path[1] == 'D':
                    unique_mode_paths[unique_path]['ExplicitModes'].append(ModeD[final_path].conj())
                elif path[1] == 'T':
                    unique_mode_paths[unique_path]['ExplicitModes'].append(ModeT[final_path].conj())
                else:
                    raise ValueError('Failed to identiy type of the Mode')
            elif path[0] == '1':
                final_path = path[3:]+'_t'+str(snktime)
                if path[1] == 'D':
                    unique_mode_paths[unique_path]['ExplicitModes'].append(ModeD[final_path])
                elif path[1] == 'T':
                    unique_mode_paths[unique_path]['ExplicitModes'].append(ModeT[final_path])
                else:
                    raise ValueError('Failed to identiy type of the Mode')
            else:
                raise ValueError('Failed to identify sink and source times')
    # Now Construc tht explicit and Perambulators
    Ps_indices         = ''
    Ps_Stacked_indices = []
    cntr = 0
    unique_cntr = 0
    for unique_path in unique_mode_paths_copy:
        all_clusters_numbers = unique_mode_paths[unique_path]['ExplicitPerambulators'].copy()
        Ps_Naive = []
        for cluster_number in all_clusters_numbers:
            Ps_indices_0, Ps_List = Perambulator_Extractor(all_perambulators = Prmbltr,
                                                         exp_prmp_container = Full_Cluster[cluster_number],
                                                         snktime = snktime, srctime = srctime)
            if cntr == 0:
                Ps_indices = Ps_indices_0
                cntr += 1
            else:
                if Ps_indices != Ps_indices_0:
                    raise ValueError('Failed to extract perambulator indices')
            Ps_Naive.append(Ps_List)
        unique_mode_paths[unique_path]['ExplicitPerambulators'] = stacker(Ps_Naive)
        if unique_cntr in stack_index_map:
            stckng_idx = stack_index_map[unique_cntr]
        else:
            print('Case with More Than 28 Combinations appears here!!')
        final_Ps_indices = ','.join([f'{stckng_idx}'+i for i in Ps_indices.split(',')])
        Ps_Stacked_indices.append(final_Ps_indices) 
        unique_cntr += 1
    return {'Ps_Stacked_Indices': Ps_Stacked_indices, 'Mode_P': unique_mode_paths, 'Unique_Paths': unique_mode_paths_copy}
#Construction:
'''
if True:
    if True:
        for full_cluster in self.clusters_with_kies:
            all_cluter_info   = full_cluster[1][1]
            all_Modes_Paths   = full_cluster[1][0]['MDT_Info']
            mode_indices      = ','.join(full_cluster[1][0]['Mode_Index_Info'])
            Mode_P_Info       = Perambulator_Mode_Handler(Full_Cluster = all_cluter_info, All_Mode_Info = all_Modes_Paths,
                                      snktime = self., srctime=self.SourceTime,
                                      Prmbltr = All_Perambulators, ModeD = ModeDoublets, ModeT = ModeTriplets)
            Unique_Mode = Mode_P_Info['Unique_Paths']
            stckd_P_idx = Mode_P_Info['Ps_Stacked_Indices']
            Mode_P_Info = Mode_P_Info['Mode_P']
            N = len(stckd_P_idx)
            i = 0
            try:
                res = torch.einsum(f'{mode_indices},{stckd_P_idx[i]}->{stckd_P_idx[i][0]}',
                                   *Mode_P_Info[Unique_Mode[i]]['ExplicitModes'],*Mode_P_Info[Unique_Mode[i]]['ExplicitPerambulators']).sum()
                for i in range(1, N):
                    res += torch.einsum(f'{mode_indices},{stckd_P_idx[i]}->{stckd_P_idx[i][0]}',
                                       *Mode_P_Info[Unique_Mode[i]]['ExplicitModes'],*Mode_P_Info[Unique_Mode[i]]['ExplicitPerambulators']).sum()
            except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                raise TypeError('That should not happen!!!!!!')
            clusters_with_kies_copy.append((full_cluster[0], res))
            print(cntrctns_cntr)
            cntrctns_cntr+=1
        return clusters_with_kies_copy, self.WT_numerical_factors
'''
def Perambulator_Mode_Handler(Full_Cluster = None, all_SG_perambulators = None, All_Mode_Info = None,
                             snktime = None, srctime=None, Mode_Unsplitted_Index = None,
                             Prmbltr = None, ModeD = None, ModeT = None, current_time=None, Hadron_Momenta = None):
    # Find all Modes, which are equivalent
    unique_mode_paths = {}
    for i, path in enumerate(All_Mode_Info):
        if tuple(path) in unique_mode_paths:
            unique_mode_paths[tuple(path)]['ExplicitPerambulators'].append(i)
        else:
            unique_mode_paths[tuple(path)] = {'ExplicitModes': [], 'ExplicitPerambulators': []}
            unique_mode_paths[tuple(path)]['ExplicitPerambulators'] = [i]
    unique_mode_paths_copy = [path for path in unique_mode_paths]
    # Now put in the explicit expressions of the modes!
    Stacked_Modes = []
    for unique_path in unique_mode_paths_copy:
        # path here corresponds to one_mode_path
        Mode_Container = []
        for path in unique_path:
            if path[0] == '0':
                final_path = path[3:]+'_t'+str(srctime)
                if path[1] == 'D':
                    Mode_Container.append(ModeD[final_path].conj())
                elif path[1] == 'T':
                    Mode_Container.append(ModeT[final_path].conj())
                else:
                    raise ValueError('Failed to identiy type of the Mode')
            elif path[0] == '1':
                final_path = path[3:]+'_t'+str(snktime)
                if path[1] == 'D':
                    Mode_Container.append(ModeD[final_path])
                elif path[1] == 'T':
                    Mode_Container.append(ModeT[final_path])
                else:
                    raise ValueError('Failed to identiy type of the Mode')
            else:
                raise ValueError('Failed to identify sink and source times')
        Stacked_Modes.append(Mode_Container)
    Stacked_Modes = stacker(Stacked_Modes)
    # Now Construc tht explicit and Perambulators
    Ps_indices         = ''
    cntr = 0
    Stacked_Ps = []
    for unique_path in unique_mode_paths_copy:
        Mode_Container = []
        all_clusters_numbers = unique_mode_paths[unique_path]['ExplicitPerambulators'].copy()
        for cluster_number in all_clusters_numbers:
            Ps_indices_0, Ps_List = Perambulator_Extractor(all_perambulators = Prmbltr, all_SG_perambulators = all_SG_perambulators,
                                                         exp_prmp_container = Full_Cluster[cluster_number],
                                                         snktime = snktime, srctime = srctime, current_time=current_time, Hadron_Momenta = Hadron_Momenta)
            if cntr == 0:
                Ps_indices = Ps_indices_0
                cntr += 1
            else:
                if Ps_indices != Ps_indices_0:
                    raise ValueError('Failed to extract perambulator indices')
            Mode_Container.append(Ps_List)
        Stacked_Ps.append(stacker(Mode_Container))
    Stacked_Ps  = stacker(Stacked_Ps)
    stckng_idx1 = stack_index_map[0]
    stckng_idx2 = stack_index_map[1]
    #Mode_Unsplitted_Index
    Mode_Indices = ','.join([f'{stckng_idx1}'+i for i in Mode_Unsplitted_Index.split(',')])
    Ps_indices = ','.join([f'{stckng_idx1}{stckng_idx2}'+i for i in Ps_indices.split(',')])
    #print(f'Mode_Indices: {Mode_Indices}')
    #print(f'Ps_indices: {Ps_indices}')
    return Ps_indices, Mode_Indices, Stacked_Ps, Stacked_Modes
    #return {'P_Idx': Ps_indices, 'M_Idx': Mode_Indices, 'P_T': Stacked_Ps, 'M_T': Stacked_Modes}