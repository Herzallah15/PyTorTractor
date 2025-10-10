# system modules
import os
import sys

# optional: falls psutil noch gebraucht wird
import psutil
https://chatgpt.com/c/68e8c33b-a02c-8325-800e-db6319fca66f
# project modules
from contractions_handler import *
from Hadron_Info_Converter import *
from Hadrontractions_Converter import *
from PyTorDefinitions import *
from PyTorTractor_SingleHadron import *
from PyTorTractor import *



Hadron1 = Hadron(File_Info_Path = 'data/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0', dlen='dlen0')
Hadron2 = Hadron(File_Info_Path = 'data/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,1), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0', dlen='dlen0')
Hadron3 = Hadron(File_Info_Path = 'data/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,2), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0', dlen='dlen0')
Hadron4 = Hadron(File_Info_Path = 'data/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0', dlen='dlen0')
Hadron5 = Hadron(File_Info_Path = 'data/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,1), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0', dlen='dlen0')
Hadron6 = Hadron(File_Info_Path = 'data/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,2), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0', dlen='dlen0')
hadrons = [Hadron1, Hadron2, Hadron3, Hadron4, Hadron5, Hadron6]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../JohnData/data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../JohnData/data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../JohnData/data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../JohnData/data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '3N3N.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
print('____')
print('Final_Total_Result')
for i, C in enumerate(res):
    print(f'<N+N+N+ BN+ BN+ BN+>(tsnk = {i}, tsrc = 0) = {C}')
