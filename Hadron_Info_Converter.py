import numpy as np
import torch
from torch import nn
import h5py
from itertools import product
from functools import reduce
from TorClasses import *
from contractions_handler import *
from Hadrontractions_Converter import *




er1 = 'In hadron you need to provide the following informations: HadronPosition, OpertorType, Flavor, Momentum, LGIrrep, Displacement'
er2 = 'OpertorType must be either meson_operators or baryon_operators'
def ddir(path, DT_Specifier = None, dlen = None):
    if np.all(path == np.zeros(3)):
        if dlen is None:
            path_final = 'ddir0'
        elif dlen == 'dlen0' or dlen == 0:
            path_final = 'ddir0_dlen0'
        else:
            raise ValueError('Hadron_Info_Converter: For dlen = 0/ dlen = dlen0, the displacement is expected to be zero.')
        return path_final, None
    if DT_Specifier == 'Triplet':
        if np.all(path[1:] == np.zeros(2)):
            path_final, Stack_I = f'ddir{path[0]}', 0
        elif np.all(np.array([path[0], path[2]]) == np.zeros(2)):
            path_final, Stack_I = f'ddir{path[1]}', 1
        elif np.all(np.array([path[0], path[1]]) == np.zeros(2)):
            path_final, Stack_I = f'ddir{path[2]}', 2
        else:
            raise ValueError('Current Verson of PyTorTractor can handle displaced triplets only of the form i00, 0i0 or 00i')
        if dlen is not None:
            if isinstance(dlen, int):
                dlen = 'dlen'+str(dlen)
            path_final = path_final + '_' + dlen
        return path_final, Stack_I
    elif DT_Specifier == 'Doublet':
        if np.all(np.array([path[0], path[2]]) == np.zeros(2)):
            path_final, Stack_I = f'ddir{path[1]}', None
        else:
            raise ValueError('Current Verson of PyTorTractor can handle displaced doublets only of the form 0i0')
        if dlen is not None:
            if isinstance(dlen, int):
                dlen = 'dlen'+str(dlen)
            path_final = path_final + '_' + dlen
        return path_final, Stack_I




#"0"=>0, "+"=> 1, "-"=>-1, "#"=>2, "="=>-2, "T"=>3, "t"=>-3
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


class Hadron:
    def __init__(self, File_Info_Path = None, Hadron_Type = None, Hadron_Position = None, Flavor = None,
          Momentum = None, LGIrrep = None, Displacement = None, dlen = None):
        self.File_Info_Path  = File_Info_Path
        self.Hadron_Type     = Hadron_Type
        self.Hadron_Position = Hadron_Position
        self.Flavor          = Flavor
        self.Momentum        = Momentum
        self.LGIrrep         = LGIrrep
        self.Displacement    = Displacement
        self.dlen            = dlen
        if None in (File_Info_Path, Hadron_Type, Hadron_Position, Flavor, Momentum, LGIrrep , Displacement):
            raise ValueError(er1)
        if Hadron_Type not in ['meson_operators', 'baryon_operators']:
            raise ValueError(er2)
    def getFile_Info_Path(self):
        return self.File_Info_Path
    def getHadron_Type(self):
        return self.Hadron_Type
    def getHadron_Position(self):
        return tuple(self.Hadron_Position)
    def getFlavor(self):
        return self.Flavor
    def getMomentum_Path(self):
        return momentum(self.Momentum)['mom_path']
    def getMomentum_Value(self):
        return momentum(self.Momentum)['int_value']
    def getGroup(self):
        return self.LGIrrep
    def getDisplacement(self):
        return self.Displacement
    def getInfo(self):
        with h5py.File(self.getFile_Info_Path(), 'r') as info_container:
            ht                   = self.getHadron_Type()
            fl, mom              = self.getFlavor(), self.getMomentum_Path()
            grp, disp            = self.getGroup(), self.getDisplacement()
            spin_structure_info  = info_container[ht][fl][mom][grp][disp]['ivals'][:]
            N_Combinations       = spin_structure_info[0]
            Spin_Displacement_N  = spin_structure_info[1]
            spin_structures      = spin_structure_info[2:].reshape(N_Combinations, Spin_Displacement_N)
            coefcients_info      = info_container[ht][fl][mom][grp][disp]['dvals'][:].reshape(N_Combinations, 2)
            coefficients         = coefcients_info[:, 0] + 1j * coefcients_info[:, 1]
        list_info = []
        Hdrn = self.getHadron_Position()
        for i in range(N_Combinations):
            c_info_i = spin_structures[i]
            q0, q1 = Hdrn + (0,), Hdrn + (1,)
            comb_i = {q0: c_info_i[0], q1: c_info_i[1], Hdrn: {'Factor': coefficients[i]} }
            if Spin_Displacement_N == 6:
                disp_i = ddir(c_info_i[-3:], DT_Specifier = 'Triplet', dlen = self.dlen)
                q2         = Hdrn + (2,)
                comb_i[q2] = c_info_i[2]
                comb_i[Hdrn]['MomDis'] = {'MT': self.getMomentum_Value()+'_'+ disp_i[0], 'dis_dir':  disp_i[1]}
            elif Spin_Displacement_N ==5:
                disp_i = ddir(c_info_i[-3:], DT_Specifier = 'Doublet', dlen = self.dlen)
                comb_i[Hdrn]['MomDis'] = {'MD': self.getMomentum_Value()+'_'+ disp_i[0], 'dis_dir':  disp_i[1]}
            else:
                raise ValueError('Failed to extract the hadron informations *')
            list_info.append(comb_i)
        return list_info

def hadron_info_multiplier(*hadrons):
    hadrons = [hadron.getInfo() for hadron in hadrons]
    def two_map_multiplier(map1, map2):
        final_info_map = copy.deepcopy(map1)
        for info in map2:
            final_info_map[info] = copy.deepcopy(map2[info])
        return final_info_map
    return [reduce(two_map_multiplier, combo) for combo in product(*hadrons)]






class ExplicitPerambulator:
    #onecomb_info is one element of hadron_info_multiplier
    def __init__(self, perambulator:Perambulator, onecomb_info):
        self.perambulator    = perambulator
        self.onecomb_info    = onecomb_info
    def getPerambulator(self):
        return self.perambulator
    def getH(self):
        return tuple(self.getPerambulator().getH().tolist())
    def getH_Bar(self):
        return tuple(self.getPerambulator().getH_Bar().tolist())
    def getQ(self):
        return tuple(self.getPerambulator().getQ().tolist())
    def getQ_Bar(self):
        return tuple(self.getPerambulator().getQ_Bar().tolist())
    def getS(self):
        return self.onecomb_info[tuple(self.getPerambulator().getQ().tolist())].item()
    def getS_Bar(self):
        return self.onecomb_info[tuple(self.getPerambulator().getQ_Bar().tolist())].item()
    def getDis(self):
        return self.onecomb_info[tuple(self.getPerambulator().getH().tolist())]['dis']
    def getDis_Bar(self):
        return self.onecomb_info[tuple(self.getPerambulator().getH_Bar().tolist())]['dis']
    def getFF_H(self):
        if self.getH()[0] == 0:
            ff1 = self.onecomb_info[tuple(self.getPerambulator().getH().tolist())]['Factor'].item().conjugate()
        elif self.getH()[0] == 1:
            ff1 = self.onecomb_info[tuple(self.getPerambulator().getH().tolist())]['Factor'].item()
        else:
            raise ValueError('A hadron can be either on sink 1 or source 0')
        return ff1
    def getFF_H_Bar(self):
        if self.getH_Bar()[0] == 0:
            ff2 = self.onecomb_info[tuple(self.getPerambulator().getH_Bar().tolist())]['Factor'].item().conjugate()
        elif self.getH_Bar()[0] == 1:
            ff2 = self.onecomb_info[tuple(self.getPerambulator().getH_Bar().tolist())]['Factor'].item()
        else:
            raise ValueError('A hadron can be either on sink 1 or source 0')
        return ff2
    def getFlavor(self):
        return self.getPerambulator().getFlvr()


class ExplicitPerambulator_Container_OneComb:
    def __init__(self, perambulator_container:Perambulator_Container, onecomb_info):
        self.perambulator_container = perambulator_container
        self.onecomb_info           = onecomb_info
    def getModeInfos(self):
        non_ex_perambulators = self.perambulator_container.getPerambulators()
        hadrons_NB    = [tuple(hadron.getH().tolist()) for hadron in non_ex_perambulators]
        hadrons_B     = [tuple(hadron.getH_Bar().tolist()) for hadron in non_ex_perambulators]
        hadrons_all   = hadrons_NB + hadrons_B
        hadrons_all   = set(hadrons_all)
        mom_dis_paths, index_list = [], []
        stack_Idxlist, stakout_index = [], ''
        for hadron in hadrons_all:
            q0, q1 = hadron + (0,), hadron + (1,)
            contraction_indices = index_map[q0] + index_map[q1]
            MomDis_Info = self.onecomb_info[hadron]['MomDis']
            stac_info = MomDis_Info['dis_dir']
            if list(MomDis_Info.keys())[0] == 'MD':
                link = str(hadron[0])+'D_'+ MomDis_Info['MD']
            elif list(MomDis_Info.keys())[0] == 'MT':
                q2 = hadron + (2,)
                link = str(hadron[0])+'T_'+ MomDis_Info['MT']
                contraction_indices = contraction_indices + index_map[q2]
            else:
                raise ValueError('Failed to identiy the Modes')
            if stac_info is not None:
                contraction_indices = MDT_index_map[hadron] + contraction_indices
                stakout_index += MDT_index_map[hadron]
            mom_dis_paths.append(link)
            index_list.append(contraction_indices)
            stack_Idxlist.append(stac_info)
        return {'mom_dis_info': mom_dis_paths, 'index_info': tuple(index_list),
                'stacked_indices': stack_Idxlist, 'stackedout_indices': stakout_index}
        #stack_Idxlist is now of the form [0, 1, None, 2,...]. It cannot made out of other numbers!
        #print({'mom_dis_info': mom_dis_paths, 'index_info': tuple(index_list)})
        # At the end we have the following informations:
        # mom_dis_info       = [path_MDT0, path_MDT1, ....]
        # index_info         = [MDT0_FullIndices, MDT1_FullIndices, ...]
        # stacked_indices    = [MDT0_Stacked_Index_Int, MDT1_Stacked_Index_Int, ...]
        # stackedout_indices = MDT0_Stacked_Index_Str+MDT1_Stacked_Index_Str+...

    def getExplicit_Perambulators(self):
        non_ex_perambulators = self.perambulator_container.getPerambulators()
        return [ExplicitPerambulator(P, self.onecomb_info) for P in non_ex_perambulators]

class Final_Perambulator_Container:
    #all_comb is the result of hadron_info_multiplier
    def __init__(self, perambulator_container:Perambulator_Container, all_comb):
        self.perambulator_container = perambulator_container
        self.all_comb               = all_comb
    def getExplicit_Perambulator_Containers(self):
        one_perambulator_container_to_all_explicit = []
        for onecomb_info in self.all_comb:
            one_perambulator_container_to_all_explicit.append(
                ExplicitPerambulator_Container_OneComb(self.perambulator_container, 
                                                       onecomb_info).getExplicit_Perambulators())
        return one_perambulator_container_to_all_explicit
    def getModeInfos(self):
        tracking_momdis = set()#This is to see we have different mode informations in one container or not!
        indices_info    = set()#We expect, for one Final_Container, i.e., a container with all possiblities, that 
                               #the indizes of the laphs do not change, since the final container is simply one conainer with
                               #all possiblie spin combinations! However, the modes themselves could change in regard of momentum/dis
        outer_StIndices ,stack_index_info, tracking_stack  = set(), [], set()
        for onecomb_info in self.all_comb:
            exp_result_mode_info = ExplicitPerambulator_Container_OneComb(self.perambulator_container, onecomb_info).getModeInfos()
            tracking_momdis.add(tuple(exp_result_mode_info['mom_dis_info']))
            indices_info.add(tuple(exp_result_mode_info['index_info']))
            outer_StIndices.add(exp_result_mode_info['stackedout_indices'])
            stack_index_info.append(exp_result_mode_info['stacked_indices'])
            tracking_stack.add(tuple(exp_result_mode_info['stacked_indices']))
        if len(indices_info) != 1:
            raise ValueError('Failed to extrac the indices of the Modes')
        indices_information = list(indices_info)[0]
        if len(tracking_momdis) != 1:
            raise ValueError('Failed to extract the pathes of the Modes')
        all_momdis_info = list(tracking_momdis)[0]
        if len(outer_StIndices) !=1:
            raise ValueError('Failed to find/order the displaced Modes for all spin/dis combinations')
        outer_StIndices = list(outer_StIndices)[0]
        if len(tracking_stack) == 1:
            if (all(item == None for item in stack_index_info[0])):
                if outer_StIndices != '':
                    ValueError('Failed to extract MDT with one displacement combination')
                    #here we should reproduced what we had until now!
                return {'Mode_Index_Info': indices_information, 'MDT_Info': all_momdis_info, 'MDT_Stack': None}
            else:
                #here we should have the same as we had till now but with extra specification of the stacked indices in the Modes!
                for i, index_str in enumerate(stack_index_info[0]):
                    if index_str is not None:
                        indices_information[i] = indices_information[i][1:]
                return {'Mode_Index_Info': indices_information, 'MDT_Info': all_momdis_info,' MDT_Stack': tuple(list(tracking_stack)[0])}
        else:
            return {'Mode_Index_Info': indices_information, 'MDT_Info': all_momdis_info,
                    'MDT_Stack': {'tr_ndcs' : outer_StIndices, 'MDT_II': stack_index_info}}