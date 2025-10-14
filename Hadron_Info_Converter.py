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
        return path_final
    if DT_Specifier == 'Triplet':
        if np.all(path[1:] == np.zeros(2)):
            path_final = f'ddir{path[0]}'
        elif np.all(np.array([path[0], path[2]]) == np.zeros(2)):
            path_final = f'ddir0_{path[1]}_0'
        elif np.all(np.array([path[0], path[1]]) == np.zeros(2)):
            path_final = f'ddir00_{path[2]}'
        else:
            raise ValueError('Current Verson of PyTorTractor can handle displaced triplets only of the form i00, 0i0 or 00i')
        if dlen is not None:
            if isinstance(dlen, int):
                dlen = 'dlen'+str(dlen)
            path_final = path_final + '_' + dlen
        return path_final
    elif DT_Specifier == 'Doublet':
        if np.all(np.array([path[0], path[2]]) == np.zeros(2)):
            path_final = f'ddir{path[1]}'
        else:
            raise ValueError('Current Verson of PyTorTractor can handle displaced doublets only of the form 0i0')
        if dlen is not None:
            if isinstance(dlen, int):
                dlen = 'dlen'+str(dlen)
            path_final = path_final + '_' + dlen
        return path_final




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
    def getMomentum(self):
        return self.Momentum
    def getMomentum_Path(self):
        return momentum(self.Momentum)['mom_path']
    def getMomentum_Value(self):
        return momentum(self.Momentum)['int_value']
    def getGroup(self):
        return self.LGIrrep
    def getDisplacement(self):
        return self.Displacement
    def getDlen(self):
        return self.dlen
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
                comb_i[Hdrn]['MomDis'] = {'MT': self.getMomentum_Value()+'_'+ disp_i}
            elif Spin_Displacement_N ==5:
                disp_i = ddir(c_info_i[-3:], DT_Specifier = 'Doublet', dlen = self.dlen)
                comb_i[Hdrn]['MomDis'] = {'MD': self.getMomentum_Value()+'_'+ disp_i}
            else:
                raise ValueError('Failed to extract the hadron informations *')
            list_info.append(comb_i)
        return list_info

    def __mul__(self, other):
        if isinstance(other, TwoHadron):
            T_Mul = {}
            for i in range(other.getN()):
                hdrn123 = other.getallCombi()[f'combi_{i}']['Hadrons'] + [self]
                ForFactor = other.getallCombi()[f'combi_{i}']['Factor']
                T_Mul[f'combi_{i}'] = {'Hadrons': hdrn123, 'Factor': ForFactor}
            return T_Mul
        elif isinstance(other, Hadron):
            return {'combi_0': {'Hadrons': [self, other], 'Factor': 1} }
        elif isinstance(other, dict):
            T_Mul = {}
            if not all(i.startswith('combi_') for i in other):
                raise TypeError('Undefined multiplicaton with a hadron object')
            for i, comb_y in enumerate(other):
                hdrns   = [self] + other[comb_y]['Hadrons']
                FFactor = other[comb_y]['Factor']
                T_Mul[f'combi_{i}'] = {'Hadrons': hdrns, 'Factor': FFactor}
            return T_Mul
    def __rmul__(self, other):
        if isinstance(other, TwoHadron):
            return self * other
        elif isinstance(other, dict):
            return self * other
        else:
            raise TypeError('Undefined multiplication with a single-hadron operator')

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
        for hadron in hadrons_all:
            q0, q1 = hadron + (0,), hadron + (1,)
            contraction_indices = index_map[q0] + index_map[q1]
            MomDis_Info = self.onecomb_info[hadron]['MomDis']
            if list(MomDis_Info.keys())[0] == 'MD':
                link = str(hadron[0])+'D_'+ MomDis_Info['MD']
            elif list(MomDis_Info.keys())[0] == 'MT':
                q2 = hadron + (2,)
                link = str(hadron[0])+'T_'+ MomDis_Info['MT']
                contraction_indices = contraction_indices + index_map[q2]
            else:
                raise ValueError('Failed to identiy the Modes')
            mom_dis_paths.append(link)
            index_list.append(contraction_indices)
        return {'mom_dis_info': mom_dis_paths, 'index_info': tuple(index_list)}
        # At the end we have the following informations:
        # mom_dis_info       = [path_MDT0, path_MDT1, ....]
        # index_info         = [MDT0_FullIndices, MDT1_FullIndices, ...]

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
        all_momdis_info = []# In case the displacement itself changes in the different combinations!
        indices_info    = set()#We expect, for one Final_Container, i.e., a container with all possiblities, that 
                               #the indizes of the laphs do not change, since the final container is simply one conainer with
                               #all possiblie spin combinations! However, the modes themselves could change in regard of momentum/dis
        for onecomb_info in self.all_comb:
            exp_result_mode_info = ExplicitPerambulator_Container_OneComb(self.perambulator_container, onecomb_info).getModeInfos()
            tracking_momdis.add(tuple(exp_result_mode_info['mom_dis_info']))
            all_momdis_info.append(exp_result_mode_info['mom_dis_info'])
            indices_info.add(tuple(exp_result_mode_info['index_info']))
        if len(indices_info) != 1:
            raise ValueError('Failed to extrac the indices of the Modes')
        indices_information = list(indices_info)[0]
        return {'Mode_Index_Info': indices_information, 'MDT_Info': all_momdis_info}

'''
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
##########     ##########
'''
##########
class TwoHadron:
    def __init__(self, File_Info_Path = None, Total_Momentum = None, LGIrrep = None, Hadron1 = None, Hadron2 = None, OpNum = None, strangeness = None):
        self.File_Info_Path = File_Info_Path
        self.Total_Momentum = Total_Momentum
        self.LGIrrep        = LGIrrep
        self.Hadron1        = Hadron1
        self.Hadron2        = Hadron2
        self.strangeness    = strangeness 
        if self.Hadron1.getHadron_Position()[0] != self.Hadron2.getHadron_Position()[0]:
            raise ValueError('Both of the hadrons must be on the same time slice!')
        if self.Hadron1.getHadron_Position()[0] not in [0, 1]:
            raise ValueError('Hadron operator must be either on sink or source')
        self.Position = self.Hadron1.getHadron_Position()[0]
        self.OpNum          = OpNum
        if OpNum is None:
            self.OpNum = '0'
        else:
            if isinstance(OpNum, (int,float)):
                self.OpNum = str(OpNum)
            else:
                self.OpNum = OpNum
        if (self.Hadron1.getHadron_Type() == 'meson_operators') and (self.Hadron2.getHadron_Type() == 'baryon_operators'):
            twohadron_type = 'MesonBaryon'
        elif (self.Hadron1.getHadron_Type() == 'baryon_operators') and (self.Hadron2.getHadron_Type() == 'baryon_operators'):
            twohadron_type = 'BaryonBaryon'
        elif (self.Hadron1.getHadron_Type() == 'meson_operators') and (self.Hadron2.getHadron_Type() == 'meson_operators'):
            twohadron_type = 'MesonMeson'
        else:
            raise ValueError('Unrecognized type of two-hadron operators. You can have MesonBaryon, BaryonBaryon or MesonMeson')
        if twohadron_type in ['MesonBaryon', 'BaryonBaryon']:
            Hierarchy_0 =  self.Hadron1.getHadron_Type().split('_')[0] + '_' + self.Hadron2.getHadron_Type().split('_')[0]+'_operators'
            Hierarchy_1 =  momentum(self.Total_Momentum)['mom_path']
            Hierarchy_2 =  self.LGIrrep
            Hierarchy_3 =  self.Hadron1.getMomentum_Path()[8:] + '_' + self.Hadron1.getGroup() + '_'
            Hierarchy_3 += self.Hadron2.getMomentum_Path()[8:] + '_' + self.Hadron2.getGroup() + '_' + self.OpNum
            self.two_H_path = '/'+ Hierarchy_0 + '/' + Hierarchy_1 + '/'+ Hierarchy_2 + '/'+ Hierarchy_3
            print(f'Path of the two hadron operator: {self.two_H_path}')
            with h5py.File(self.File_Info_Path, 'r') as yunus0:
                if self.two_H_path not in yunus0:
                    raise KeyError(f'The path {self.two_H_path} is not found in {self.File_Info_Path}')
                yunus = yunus0[self.two_H_path]
                self.N = yunus['ivals'][:][0]#Number of vertical   combinations
                M = yunus['ivals'][:][1]#Number of horizontal combinations
                Numerical_Coefficients = yunus['dvals'][:].reshape(self.N, 2)
                Numerical_Coefficients = Numerical_Coefficients[:,0] + 1j * Numerical_Coefficients[:, 1]
                if self.Position == 0:
                    self.Numerical_Coefficients = Numerical_Coefficients.conj()
                else:
                    self.Numerical_Coefficients = Numerical_Coefficients
                self.Hadron_TotalCombi      = yunus['ivals'][2:][:].reshape(self.N, M)
            T = {}
            for i in range(self.N):
                H1_Momentum = tuple(self.Hadron_TotalCombi[i][0:3].tolist())
                H1_Group = self.Hadron1.getGroup() + '_' + str(self.Hadron_TotalCombi[i][3])
                
                H2_Momentum = tuple(self.Hadron_TotalCombi[i][4:7].tolist())
                H2_Group = self.Hadron2.getGroup() + '_' + str(self.Hadron_TotalCombi[i][7])
                
                hdrn1 = Hadron(File_Info_Path = self.Hadron1.getFile_Info_Path(), Hadron_Type = self.Hadron1.getHadron_Type(),
                               Hadron_Position = self.Hadron1.getHadron_Position(), Flavor = self.Hadron1.getFlavor(),
                               Momentum = H1_Momentum, LGIrrep = H1_Group, 
                               Displacement = self.Hadron1.getDisplacement(), dlen = self.Hadron1.getDlen())
                hdrn2 = Hadron(File_Info_Path = self.Hadron2.getFile_Info_Path(), Hadron_Type = self.Hadron2.getHadron_Type(),
                               Hadron_Position = self.Hadron2.getHadron_Position(), Flavor = self.Hadron2.getFlavor(),
                               Momentum = H2_Momentum, LGIrrep = H2_Group, 
                               Displacement = self.Hadron2.getDisplacement(), dlen = self.Hadron2.getDlen())
                ForFactor = self.Numerical_Coefficients[i]
                T[f'combi_{i}'] = {'Hadrons': [hdrn1, hdrn2], 'Factor': ForFactor}
            self.alIn = T
        elif twohadron_type == 'MesonMeson':
            if self.strangeness is None:
                raise ValueError('For MesonMeson operators you must specifiy the strangeness!')
            def flavor_specify(strngnss, hdrn_flvr):
                if strngnss == 1:
                    return 'kaon_su'
                elif strngnss == -1:
                    return 'antikaon_ds'
                elif strngnss == 0:
                    return hdrn_flvr
                else:
                    raise ValueError('Strangeness can be either 1, -1 or 0')
            Hierarchy_0  =  self.Hadron1.getHadron_Type().split('_')[0] + '_' + self.Hadron2.getHadron_Type().split('_')[0]+'_operators'
            Hierarchy_1  =  momentum(self.Total_Momentum)['mom_path']
            Hierarchy_2  =  self.LGIrrep
            Hierarchy_3  =  'S=' + str(self.strangeness) + '_'
            Hierarchy_3 +=  self.Hadron1.getMomentum_Path()[8:] + '_' + self.Hadron1.getGroup() + '_'
            Hierarchy_3 += self.Hadron2.getMomentum_Path()[8:] + '_' + self.Hadron2.getGroup() + '_' + self.OpNum
            self.two_H_path = '/'+ Hierarchy_0 + '/' + Hierarchy_1 + '/'+ Hierarchy_2 + '/'+ Hierarchy_3
            print(f'Path of the two hadron operator: {self.two_H_path}')
            with h5py.File(self.File_Info_Path, 'r') as yunus0:
                if self.two_H_path not in yunus0:
                    raise KeyError(f'The path {self.two_H_path} is not found in {self.File_Info_Path}')
                yunus = yunus0[self.two_H_path]
                self.N = yunus['ivals'][:][0]#Number of vertical   combinations
                M = yunus['ivals'][:][1]#Number of horizontal combinations
                Numerical_Coefficients = yunus['dvals'][:].reshape(self.N, 2)
                Numerical_Coefficients = Numerical_Coefficients[:,0] + 1j * Numerical_Coefficients[:, 1]
                if self.Position == 0:
                    self.Numerical_Coefficients = Numerical_Coefficients.conj()
                else:
                    self.Numerical_Coefficients = Numerical_Coefficients
                self.Hadron_TotalCombi      = yunus['ivals'][2:][:].reshape(self.N, M)
            T = {}
            for i in range(self.N):
                H1_Momentum = tuple(self.Hadron_TotalCombi[i][0:3].tolist())
                H1_Flavor   = flavor_specify(self.Hadron_TotalCombi[i][3], self.Hadron1.getFlavor())
                H1_Group = self.Hadron1.getGroup() + '_' + str(self.Hadron_TotalCombi[i][4])
                
                H2_Momentum = tuple(self.Hadron_TotalCombi[i][5:8].tolist())
                H2_Flavor   = flavor_specify(self.Hadron_TotalCombi[i][8], self.Hadron2.getFlavor())
                H2_Group = self.Hadron2.getGroup() + '_' + str(self.Hadron_TotalCombi[i][9])
                
                hdrn1 = Hadron(File_Info_Path = self.Hadron1.getFile_Info_Path(), Hadron_Type = self.Hadron1.getHadron_Type(),
                               Hadron_Position = self.Hadron1.getHadron_Position(), Flavor = H1_Flavor,
                               Momentum = H1_Momentum, LGIrrep = H1_Group, 
                               Displacement = self.Hadron1.getDisplacement(), dlen = self.Hadron1.getDlen())
                hdrn2 = Hadron(File_Info_Path = self.Hadron2.getFile_Info_Path(), Hadron_Type = self.Hadron2.getHadron_Type(),
                               Hadron_Position = self.Hadron2.getHadron_Position(), Flavor = H2_Flavor,
                               Momentum = H2_Momentum, LGIrrep = H2_Group, 
                               Displacement = self.Hadron2.getDisplacement(), dlen = self.Hadron2.getDlen())
                ForFactor = self.Numerical_Coefficients[i]
                T[f'combi_{i}'] = {'Hadrons': [hdrn1, hdrn2], 'Factor': ForFactor}
            self.alIn = T
        else:
            raise ValueError(f' Unrecognized type of two-hadron operator {twohadron_type}')
    def getN(self):
        return self.N
    def getallCombi(self):
        return self.alIn
    def __mul__(self, other):
        if isinstance(other, Hadron):
            T_Mul = {}
            for i in range(self.N):
                hdrn123 = self.alIn[f'combi_{i}']['Hadrons'] + [other]
                ForFactor = self.alIn[f'combi_{i}']['Factor']
                T_Mul[f'combi_{i}'] = {'Hadrons': hdrn123, 'Factor': ForFactor}
            return T_Mul
        elif isinstance(other, TwoHadron):
            T_Mul = {}
            counter = 0
            for comb_x in self.getallCombi():
                for comb_y in other.getallCombi():
                    hdrns = self.getallCombi()[comb_x]['Hadrons'] + other.getallCombi()[comb_y]['Hadrons']
                    FFactor = self.getallCombi()[comb_x]['Factor'] * other.getallCombi()[comb_y]['Factor']
                    T_Mul[f'combi_{counter}'] = {'Hadrons': hdrns, 'Factor': FFactor}
                    counter += 1
            return T_Mul
        elif isinstance(other, dict):
            if not all(i.startswith('combi_') for i in other):
                raise TypeError('Undefined multiplicaton with a TwoHadron object')
            T_Mul = {}
            counter = 0
            for comb_x in self.getallCombi():
                for comb_y in other:
                    hdrns = self.getallCombi()[comb_x]['Hadrons'] + other[comb_y]['Hadrons']
                    FFactor = self.getallCombi()[comb_x]['Factor'] * other[comb_y]['Factor']
                    T_Mul[f'combi_{counter}'] = {'Hadrons': hdrns, 'Factor': FFactor}
                    counter += 1
            return T_Mul
    def __rmul__(self, other):
        if isinstance(other, Hadron):
            return self * other
        elif isinstance(other, dict):
            return self * other
        else:
            raise TypeError('Undefined multiplication with a two-hadron operator')