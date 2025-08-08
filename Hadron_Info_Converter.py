import numpy as np
import torch
from torch import nn
import h5py
from itertools import product
from functools import reduce
from TorClasses import *
from contractions_handler import *
from Hadrontractions_Converter import *


#(FileName = 'OpInfo/baryon_operators.h5', HadronPosition = (1,0), OpertorType = 'baryon_operators', Flavor = 'delta_uud',
#Momentum = 'mom_ray_000', LGIrrep = 'Hg_1', Displacement = 'SS_0')

er1 = 'In hadron you need to provide the following informations: HadronPosition, OpertorType, Flavor, Momentum, LGIrrep, Displacement'
er2 = 'OpertorType must be either meson_operators or baryon_operators'

class Hadron:
    def __init__(self, File_Info_Path = None, Hadron_Type = None, Hadron_Position = None, Flavor = None,
          Momentum = None, LGIrrep = None, Displacement = None):
        self.File_Info_Path  = File_Info_Path
        self.Hadron_Type     = Hadron_Type
        self.Hadron_Position = Hadron_Position
        self.Flavor          = Flavor
        self.Momentum        = Momentum
        self.LGIrrep         = LGIrrep
        self.Displacement    = Displacement
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
    def getGroup(self):
        return self.LGIrrep
    def getDisplacement(self):
        return self.Displacement
    def getInfo(self):
        with h5py.File(self.getFile_Info_Path(), 'r') as info_container:
            ht                   = self.getHadron_Type()
            fl, mom              = self.getFlavor(), self.getMomentum()
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
            comb_i = {q0: c_info_i[0], q1: c_info_i[1], Hdrn: {'Factor': coefficients[i], 'dis': c_info_i[-3:]} }
            if Spin_Displacement_N == 6:
                q2         = Hdrn + (2,)
                comb_i[q2] = c_info_i[1]
            list_info.append(comb_i)
        return list_info

def hadron_info_multiplier(*hadrons):
    hadrons = [hadron.getInfo() for hadron in hadrons]
    def two_map_multiplier(map1, map2):
        final_info_map = map1.copy()
        for info in map2:
            final_info_map[info] = map2[info]
        return final_info_map
    return [reduce(two_map_multiplier, combo) for combo in product(*hadrons)]
     





class ExplicitPerambulator:
    #onecomb_info is one element of hadron_info_multiplier
    def __init__(self, perambulator:Perambulator, onecomb_info):
        self.perambulator = perambulator
        self.onecomb_info    = onecomb_info
    def getPerambulator(self):
        return self.perambulator
    def getS(self):
        return self.onecomb_info[tuple(self.getPerambulator().getQ().tolist())].item()
    def getS_Bar(self):
        return self.onecomb_info[tuple(self.getPerambulator().getQ_Bar().tolist())].item()
    def getDis(self):
        return self.onecomb_info[tuple(self.getPerambulator().getH().tolist())]['dis']
    def getDis_Bar(self):
        return self.onecomb_info[tuple(self.getPerambulator().getH_Bar().tolist())]['dis']
    def getFF(self):
        ff1 = self.onecomb_info[tuple(self.getPerambulator().getH().tolist())]['Factor'].item()
        ff2 = self.onecomb_info[tuple(self.getPerambulator().getH_Bar().tolist())]['Factor'].item()
        return ff1 * ff2



class ExplicitPerambulator_Container_OneComb:
    def __init__(self, perambulator_container:Perambulator_Container, onecomb_info):
        self.perambulator_container = perambulator_container
        self.onecomb_info           = onecomb_info
    def getEPerambulator_Container(self):
        non_ex_perambulators = self.perambulator_container.getPerambulators()
        return [ExplicitPerambulator(P, self.onecomb_info) for P in non_ex_perambulators]