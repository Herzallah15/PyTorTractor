# system modules
import os
import sys

# optional: falls psutil noch gebraucht wird
#import psutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Source")))
# project modules
from contractions_handler import *
from Hadron_Info_Converter import *
from Hadrontractions_Converter import *
from PyTorDefinitions import *
from PyTor_S_Definitions import *
from PyTorTractor_SingleHadron import *
from PyTorTractor import *
import time
def compare_c_results(res, C_Correct, epsilon):
    for i in range(len(res)):
        diff = res[i] - C_Correct[i]
        print(f'C(t_snk= {i}, t_src=0) = {diff}')
        if np.abs(diff.imag) > epsilon or np.abs(diff.real) > epsilon:
            print('Error !!!!!!!!!!')
            raise ValueError('Failed to reproduce correct results for')
        
print('Generate All Wick Diagrams')
#<Corr> {eta P=(0,0,0) A1gp_1 SS_0} {eta P=(0,0,0) A1gp_1 SS_0}</Corr>
O1 = Phi()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Phi())
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/Phi_Phi.hdf5')



#<Corr> {eta P=(0,0,0) A1gp_1 SS_0} {eta P=(0,0,0) A1gp_1 SS_0}</Corr>
Eta_Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isoscalar',
          Momentum = (0,0,0), LGIrrep = 'A1gp_1', Displacement = 'SS_0')


#{eta P=(0,0,0) A1gp_1 SS_0}</Corr>
Eta_Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isoscalar',
          Momentum = (0,0,0), LGIrrep = 'A1gp_1', Displacement = 'SS_0')



hadrons = [Eta_Sink, Eta_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []

for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/Phi_Phi.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
C_Correct = [98.54868641594737+7.65227101438224e-11j, 104.97502175339913+7.088756911061619e-11j, 105.14795330226421+4.438515664059283e-11j,
            105.18366966136358+5.456269542331393e-11j, 105.50896017109444+1.059018525889477e-10j, 105.06986267204138+1.620066300822612e-11j,
            105.42302149318428+1.9286957467925206e-11j, 105.47895661882694+4.806713975385294e-11j]
compare_c_results(res, C_Correct, 1e-10)





#<Corr> {eta P=(0,0,0) A1gp_1 SS_0} {eta P=(0,0,0) A1gp_1 SS_0}</Corr>
O1 = Eta()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Eta())
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/Eta_Eta.hdf5')


#<Corr> {eta P=(0,0,0) A1gp_1 SS_0} {eta P=(0,0,0) A1gp_1 SS_0}</Corr>
Eta_Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isoscalar',
          Momentum = (0,0,0), LGIrrep = 'A1gp_1', Displacement = 'SS_0')


#{eta P=(0,0,0) A1gp_1 SS_0}</Corr>
Eta_Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isoscalar',
          Momentum = (0,0,0), LGIrrep = 'A1gp_1', Displacement = 'SS_0')



hadrons = [Eta_Sink, Eta_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []

for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/Eta_Eta.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
C_Correct = [132.39189185908862-6.333793511804741e-09j, 136.51374855405973-3.8065463040420775e-09j, 136.71275637749184-9.884951059063479e-10j,
            136.77158276216605+1.2249840912554402e-09j, 137.08859004569396-3.2761914042769014e-09j, 136.64830424274322-3.26803368363933e-09j,
            137.0122732678159-3.1700470063868213e-09j, 137.03255681410414-3.2775910121134947e-09j]
compare_c_results(res, C_Correct, 1e-10)





#<Corr> {eta P=(0,0,0) A1gp_1 SS_0} {phi P=(0,0,0) A1gp_1 SS_0}</Corr> 
O1 = Eta()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Phi())
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/Eta_Phi.hdf5')

#<Corr> {eta P=(0,0,0) A1gp_1 SS_0}
Eta_Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isoscalar',
          Momentum = (0,0,0), LGIrrep = 'A1gp_1', Displacement = 'SS_0')


#{phi P=(0,0,0) A1gp_1 SS_0}</Corr>
Phi_Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isoscalar',
          Momentum = (0,0,0), LGIrrep = 'A1gp_1', Displacement = 'SS_0')



hadrons = [Eta_Sink, Phi_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []

for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/Eta_Phi.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
C_Correct = [119.83886085634333-2.8199397786486077e-09j, 119.71329499779587-4.4543618635524313e-10j, 119.8899007152172+2.046315330823439e-09j,
            119.94177002000936+3.9882791304939326e-09j, 120.21979203507547+4.817492920000596e-11j, 119.83368476236292+4.598185876753465e-11j,
            120.15286731189511+1.3964547434923993e-10j, 120.17065497640175+4.5778381519161247e-11j]
compare_c_results(res, C_Correct, 1e-10)





#<Corr> {eta P=(0,0,0) A1gp_1 SS_0} {isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SD_1]}</Corr>
O1 = Eta()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1/2,1/2), I=0, I3=0, A=Kaon, B=KaonC))#bar((Kaon(1/2) * KaonC(-1/2) - Kaon(-1/2) * KaonC(1/2))/(np.sqrt(2)))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/EtaIsosingletKKc.hdf5')

#<Corr> {eta P=(0,0,0) A1gp_1 SS_0}
Eta_Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isoscalar',
          Momentum = (0,0,0), LGIrrep = 'A1gp_1', Displacement = 'SS_0')


# {isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SD_1]}</Corr>
Kaon1Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Kaon2Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SD_1')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'A1gp_1', Hadron1 = Kaon1Source, Hadron2 = Kaon2Source, OpNum = 0, strangeness = 0)


hadrons = [Eta_Source, Two_Hadron_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []

for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/EtaIsosingletKKc.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
C_Correct = [-2.978403755452084e-11+0.6135527167610386j, -4.849786952511748e-11+0.5887037229850475j, -5.6560908693746755e-11+0.5869620632716133j,
            -6.622426249687421e-11+0.5870272535195734j, -4.827419792575484e-11+0.5883795107841643j, -4.842478200654983e-11+0.5864892437932011j,
            -4.911394397731292e-11+0.5880513283118514j, -4.868747484169933e-11+0.5881383775308205j]
compare_c_results(res, C_Correct, 1e-15)




#<Corr> {phi P=(0,0,0) A1gp_1 SS_0} {isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SD_1]}</Corr> 
O1 = Phi()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1/2,1/2), I=0, I3=0, A=Kaon, B=KaonC))#bar((Kaon(1/2) * KaonC(-1/2) - Kaon(-1/2) * KaonC(1/2))/(np.sqrt(2)))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/PhiIsosingletKKc.hdf5')

#<Corr> {phi P=(0,0,0) A1gp_1 SS_0}
Phi_Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isoscalar',
          Momentum = (0,0,0), LGIrrep = 'A1gp_1', Displacement = 'SS_0')


# {isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SD_1]}</Corr>
Kaon1Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Kaon2Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SD_1')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'A1gp_1', Hadron1 = Kaon1Source, Hadron2 = Kaon2Source, OpNum = 0, strangeness = 0)


hadrons = [Phi_Source, Two_Hadron_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []

for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/PhiIsosingletKKc.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

C_Correct = [-4.1426137856442197e-11+0.370962102001785j, -4.244837311783312e-11+0.5027205342368969j, -4.278632828624059e-11+0.5134393688267886j,
            -4.271823201434672e-11+0.5146988418643472j, -4.3061353921102115e-11+0.5163703745440167j, -4.245107778212513e-11+0.5142301879776356j,
            -4.2614673020826416e-11+0.515960305355448j, -4.278336574446186e-11+0.5162342808739606j]

compare_c_results(res, C_Correct, 1e-15)



# <Corr>{isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SS_0]}
Kaon1Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Kaon2Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'A1gp_1', Hadron1 = Kaon1Sink, Hadron2 = Kaon2Sink, OpNum = 0, strangeness = 0)

#{isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SD_1]}</Corr>
Kaon1Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Kaon2Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SD_1')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'A1gp_1', Hadron1 = Kaon1Source, Hadron2 = Kaon2Source, OpNum = 0, strangeness = 0)


hadrons = [Two_Hadron_Sink, Two_Hadron_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []

for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/isosingletKKC_isosingletKKC.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

C_Correct = [5.804977410762448e-11-0.9905649426644102j, 6.304388132686526e-11-0.7430870271054033j, 7.463218290712417e-11-0.7534406716817974j,
            8.502356414405443e-11-0.7553526792984477j, 6.279443807923231e-11-0.7599668137529909j, 6.206993484459116e-11-0.7542293948590948j,
            6.29405004677619e-11-0.7588340904240408j, 6.286188186470625e-11-0.7586335117678984j]
compare_c_results(res, C_Correct, 1e-15)




# <Corr>{isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SD_1]}
Kaon1Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Kaon2Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SD_1')

Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'A1gp_1', Hadron1 = Kaon1Sink, Hadron2 = Kaon2Sink, OpNum = 0, strangeness = 0)

#{isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SS_0]}</Corr>
Kaon1Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Kaon2Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'A1gp_1', Hadron1 = Kaon1Source, Hadron2 = Kaon2Source, OpNum = 0, strangeness = 0)


hadrons = [Two_Hadron_Sink, Two_Hadron_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []

for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/isosingletKKC_isosingletKKC.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
C_Correct = [-5.805000726530167e-11+0.9905649426644103j, 4.434194092635402e-11+0.6955200432396688j, 3.5623161087356813e-10+0.6750746126429328j,
            -2.8878922755763703e-10+0.7966332383246406j, -2.804147972851334e-12+0.9493934765504958j, 3.710731304477563e-11+0.9069595874222638j,
            -3.2033726406950646e-10+0.9510465716663568j, -2.2933972429473037e-12+0.7187443729244017j]
compare_c_results(res, C_Correct, 1e-15)



#<Corr>  {isosinglet_kaon_kbar A1p_1 [P=(0,0,0) A1u SS_0] [P=(0,0,1) A2 SS_0]}
# {isosinglet_kaon_kbar A1p_1 (P=(0,0,0) A1u SS_0) (P=(0,0,1) A2 SS_0)} 
overall_Group_Sink = 'A1p_1'
K1_Sink_D = 'SS_0'
K2_Sink_D = 'SS_0'

Kaon1Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = K1_Sink_D)

Kaon2Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,1), LGIrrep = 'A2', Displacement = K2_Sink_D)

Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,1), LGIrrep = overall_Group_Sink, Hadron1 = Kaon1Sink, Hadron2 = Kaon2Sink, OpNum = 0, strangeness = 0)


#{isosinglet_kaon_kbar A1p_1 [P=(0,0,0) A1u SS_0] [P=(0,0,1) A2 SS_0]}</Corr>

overall_Group_Source = 'A1p_1'
K1_Source_D = 'SS_0'
K2_Source_D = 'SS_0'

Kaon1Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = K1_Source_D)

Kaon2Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,1), LGIrrep = 'A2', Displacement = K2_Source_D)

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,1), LGIrrep = overall_Group_Source, Hadron1 = Kaon1Source, Hadron2 = Kaon2Source, OpNum = 0, strangeness = 0)


hadrons = [Two_Hadron_Sink, Two_Hadron_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res_old = [-8.501002781683127+5.91917500111841e-10j, 0.048129038660646765+0.01322920998099882j,0.003582941848136321+0.0007593832151998142j,
      0.0007177690459509104+0.0008656313030283827j, 7.241890496088926e-05+6.239618694743808e-05j, 7.830051537597468e-05+2.1371960325923867e-05j,
      -8.984523851999366e-07-2.7218472419943665e-05j, -1.7825924493434822e-05-1.6380962400032936e-05j]
res = []
for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/isosingletKKC_isosingletKKC.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

C_Correct = [-5.22891374128567+3.329904957809195e-10j, 0.00019653499550432128+2.817835722454618e-05j, 3.1325442107370044e-06+6.294061415815038e-07j,
          9.96999431582431e-08+6.268487257924901e-08j, 2.054315682831334e-07-1.6880791605007606e-07j, -2.067151783615779e-07-3.221457698130866e-08j,
          3.107397629492803e-08+2.377619916426536e-09j, 1.414162555645926e-07+7.229352760239464e-09j]
compare_c_results(res, C_Correct, 1e-15)



O1         = twoHO(rep=(1/2,1/2), I=0, I3=0, A=Kaon, B=KaonC)#(Kaon(1/2) * KaonC(-1/2) - Kaon(-1/2) * KaonC(1/2))/(np.sqrt(2))
O1_ontime  = OpTimeSlice(1, O1)
O2         = bar(O1)
O2_ontime  = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result     = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/isosingletKKC_isosingletKKC.hdf5')


#<Corr>{isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SS_0]}

overall_Group_Sink = 'A1gp_1'
K1_Sink_D = 'SS_0'
K2_Sink_D = 'SS_0'

Kaon1Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = K1_Sink_D)

Kaon2Sink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = K2_Sink_D)

Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = overall_Group_Sink, Hadron1 = Kaon1Sink, Hadron2 = Kaon2Sink, OpNum = 0, strangeness = 0)


#{isosinglet_kaon_kbar A1gp_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) A1u SS_0]}</Corr>

overall_Group_Source = 'A1gp_1'
K1_Source_D = 'SS_0'
K2_Source_D = 'SS_0'

Kaon1Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = K1_Source_D)

Kaon2Source = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = K2_Source_D)

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = overall_Group_Source, Hadron1 = Kaon1Source, Hadron2 = Kaon2Source, OpNum = 0, strangeness = 0)


hadrons = [Two_Hadron_Sink, Two_Hadron_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128, verbose=True)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []

for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/isosingletKKC_isosingletKKC.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
C_Correct = [317.57630092885347-1.4805513723758446e-08j, 225.70765460824865-5.905147820793596e-09j, 226.02948178949086-1.2362634876347183e-09j,
            226.2760982920322+2.1101968342133637e-09j, 227.63487245660107-5.148252471085747e-09j, 225.91314153310398-5.318387287225216e-09j,
            227.29178393118787-5.24455053232671e-09j, 227.23167084929355-5.2745312640024766e-09j]

compare_c_results(res, C_Correct, 1e-15)


O1 = twoHO(rep=(1/2,1), I=1/2, I3=1/2, A=Kaon, B=Pion)#(Kaon(1/2) * Pion(0) - np.sqrt(2) * Kaon(-1/2) * Pion(1))/(np.sqrt(3))
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/KPKPDoublet.hdf5')


#<Corr>{isodoublet_kaon_pion T1u_3 [P=(0,0,1) A2 SS_1] [P=(0,0,-1) A2m SS_1]}

overall_Group_Sink = 'T1u_3'
K_Sink_D = 'SS_1'
P_Sink_D = 'SS_1'

KaonSink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,1), LGIrrep = 'A2', Displacement = K_Sink_D)

PionSink = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = P_Sink_D)



Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = overall_Group_Sink, Hadron1 = KaonSink, Hadron2 = PionSink, OpNum = 0, strangeness = 1)


#{isodoublet_kaon_pion T1u_3 [P=(0,0,1) A2 SS_0] [P=(0,0,-1) A2m SS_1]}</Corr>

overall_Group_Source = 'T1u_3'
K_Source_D = 'SS_0'
P_Source_D = 'SS_1'

KaonSource = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,1), LGIrrep = 'A2', Displacement = K_Source_D)

PionSource = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = P_Source_D)



Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = overall_Group_Source, Hadron1 = KaonSource, Hadron2 = PionSource, OpNum = 0, strangeness = 1)

hadrons = [Two_Hadron_Sink, Two_Hadron_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/KPKPDoublet.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
C_Correct = [-0.006926200886367124-1.1104954536545342e-05j, 0.00016660285481550664-2.4549040891840983e-07j,
                  2.477298336814142e-05-1.4174725512434576e-06j, 6.672621799351774e-08+5.7137083115146414e-08j,
                  3.726573679872199e-08+2.1088852103782808e-08j, 3.779007969599635e-08+2.3050805020773547e-09j,
                  4.05752419933143e-09-2.01161409285776e-10j, 8.690860974711219e-10+1.2278147642880675e-10j]
compare_c_results(res, C_Correct, 1e-15)


O1 = twoHO(rep=(1/2,1), I=1/2, I3=1/2, A=Kaon, B=Pion)#(Kaon(1/2) * Pion(0) - (2**0.5) * Kaon(-1/2) * Pion(1))/(3**0.5)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Kaon(1/2))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/IsoDoubletKP_K.hdf5')

# <Corr>{isodoublet_kaon_pion T1u_3 [P=(0,0,1) A2 SS_0] [P=(0,0,-1) A2m SS_1]} {kaon P=(0,0,0) T1u_3 SS_0}</Corr>
K_Sink_D = 'SS_0'
P_Sink_D = 'SS_1'

K_Source_D = 'SS_0'




hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,1), LGIrrep = 'A2', Displacement = K_Sink_D)
hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = P_Sink_D)


hadron3 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'T1u_3', Displacement = K_Source_D)



Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'T1u_3', Hadron1 = hadron1, Hadron2 = hadron2, OpNum = 0, strangeness = 1)





hadrons = [Two_Hadron_Sink, hadron3]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/IsoDoubletKP_K.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

C_Correct_R = [-0.3115280262457777, -8.9095734805049e-05, -8.889725828771343e-06, -2.1107168104814202e-06, -1.4621830259176822e-07,
               -1.302995831228027e-08, -4.710938134005819e-09, -8.413515177672486e-09]
C_Correct_I = [-0.00011753597518867827, -2.867401201526373e-07, -1.2188494519900348e-08, 2.6123754762899825e-08,
               -3.4752794693871596e-09, 3.5610605582408835e-09, 3.697607767438184e-10, 2.8495602561701806e-11]
C_Correct = [C_Correct_R[i] + 1j*C_Correct_I[i] for i in range(len(res))]
compare_c_results(res, C_Correct, 1e-15)



O1 = Kaon(1/2)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1/2,1), I=1/2, I3=1/2, A=Kaon, B=Pion))#bar( (Kaon(1/2) * Pion(0) - (2**0.5) * Kaon(-1/2) * Pion(1))/(3**0.5))

O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/K_IsoDoubletKP.hdf5')

# <Corr>{kaon P=(0,0,0) T1u_3 SS_0} {isodoublet_kaon_pion T1u_3 [P=(0,0,1) A2 SS_0] [P=(0,0,-1) A2m SS_1]}</Corr>
#K_Sink_D = 'SS_0'
#K_Source_D = 'SS_0'
#P_Source_D = 'SS_1'



#<Corr>{kaon P=(0,0,0) T1u_3 SS_0} {isodoublet_kaon_pion T1u_3 [P=(0,0,1) A2 SS_1] [P=(0,0,-1) A2m SS_1]}</Corr>
K_Sink_D = 'SS_0'
K_Source_D = 'SS_1'
P_Source_D = 'SS_1'

hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'T1u_3', Displacement = K_Sink_D)

hadron3 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,1), LGIrrep = 'A2', Displacement = K_Source_D)
hadron4 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = P_Source_D)


Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'T1u_3', Hadron1 = hadron3, Hadron2 = hadron4, OpNum = 0, strangeness = 1)





hadrons = [hadron1, Two_Hadron_Source]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/K_IsoDoubletKP.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

    
Correct_001 = [-0.3115379976421772-0.00011345338774149455j, -8.164843689989596e-05+2.0764018967752726e-06j,
              -1.2173395558980641e-05+7.018394435943155e-07j, -3.6697915817153954e-06+8.295601370466524e-08j,
              -7.008192487681827e-07-5.846757035165485e-09j, -2.637393426280147e-08+1.3180698786244603e-09j,
              -7.852606064896001e-09+4.649337771494858e-10j, -1.5781756347592024e-08+1.4547331362227284e-10j]
Correct_011 = [-0.013753829040436115-1.0202384287617153e-05j, -8.226990824303017e-07+4.312758052081642e-08j,
              -8.187020143704914e-07-1.0762820493528314e-07j, -1.477320097173959e-08-3.0993928980057557e-09j,
              -1.8773068726664265e-08+4.427879844893765e-09j, -1.8309484800092928e-11-3.138702234717326e-10j,
              1.1601299447272747e-09+6.579975721413174e-10j, 1.7836904729872594e-10+2.5455223709267626e-10j]
compare_c_results(res, Correct_011, 1e-15)
O1 = Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = twoHO(rep=(1,1), I=1, I3=1, A=Pion, B=Pion)#TwoHadronAnnihilation(rep=(1,1), I=1, I3=1, A=Pion, B=Pion)
O2 = bar(O2)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/P_IsotripletPP.hdf5')


#<Corr>{pion P=(0,0,0) T1up_3 SS_0}
#{isotriplet_pion_pion T1up_3 [P=(0,0,1) A2m SS_1] [P=(0,0,-1) A2m SS_1]}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'T1up_3', Displacement = 'SS_0')

hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isovector_du',
          Momentum = (0,0,1), LGIrrep = 'A2m', Displacement = 'SS_1')
hadron3 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = 'SS_1')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'T1up_3', Hadron1 = hadron2, Hadron2 = hadron3, OpNum = 0, strangeness = 0)
hadrons = [Hadron1, Two_Hadron_Source]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/P_IsotripletPP.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [-0.01705939621432921+2.6603397858967667e-12j, -0.0014696825931435242+2.0855574313806832e-13j,
                    -0.00012465409755524717+4.782099150045087e-13j, -1.0373391320468922e-05+8.273672765877032e-14j,
                    -4.5396166414579006e-07+7.547742714231752e-15j, -3.658885046922376e-08+7.732876916739392e-15j,
                    -4.755738142598925e-09+4.927950734076989e-15j, -5.830128321273885e-10+1.575373692106411e-15j]
compare_c_results(res, correct_results, 1e-15)

O1 = twoHO(rep=(1/2,1), I=3/2, I3=3/2, A=Kaon, B=Pion)#Kaon(1/2) * Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/Isoquartetkpkp.hdf5')
#{isoquartet_kaon_pion A1g_1 (P=(0,0,0) A1u SS_0) (P=(0,0,0) A1um SS_0)}
#{isoquartet_kaon_pion A1g_1 (P=(0,0,0) A1u SS_0) (P=(0,0,0) A1um SS_0)
hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,1), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'A1um', Displacement = 'SS_0')

hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

hadron3 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'A1um', Displacement = 'SS_0')

hadron4 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'A1g_1', Hadron1 = hadron2, Hadron2 = hadron1, OpNum = 0, strangeness = 1)

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'A1g_1', Hadron1 = hadron4, Hadron2 = hadron3, OpNum = 0, strangeness = 1)
hadrons = [Two_Hadron_Sink, Two_Hadron_Source]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/Isoquartetkpkp.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

C_Correct = [21.318803034181443-1.5108293924085132e-09j, 0.05351444945823623-4.545121825367166e-06j, 0.0005498503231387984+1.3977208279998974e-07j,
            7.2466246077680245e-06-1.0060250590859538e-10j, 9.51895770661978e-08-4.6030702788864397e-10j, 1.005133923025637e-09-6.795685476003659e-12j,
            2.044031686822497e-11-3.2509038838690615e-13j, 4.287702244124225e-13+9.188989922362505e-16j]
compare_c_results(res, C_Correct, 1e-15)


#I = 0, I_3 = 0
O1 = Lambda()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1/2,1/2), I=0, I3=0, A=KaonC, B=Nucleon))#(bar(Nucleon(1/2)) * bar(KaonC(-1/2)) - bar(Nucleon(-1/2)) * bar(KaonC(1/2)))/(2**(1/2))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/SingletLKN.hdf5')

#<Corr>{lambda P=(0,0,0) G1u_1 SS_0} {isosinglet_kbar_nucleon G1u_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) G1g SS_0]}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'lambda_uds',
          Momentum = (0,0,0), LGIrrep = 'G1u_1', Displacement = 'SS_0', dlen='dlen0')


hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')
hadron3 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,1), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g', Displacement = 'SS_0', dlen='dlen0')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_baryon_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'G1u_1', Hadron1 = hadron2, Hadron2 = hadron3, OpNum = 0, strangeness = 1)

hadrons = [Hadron1, Two_Hadron_Source]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128,
                                     Use_Triplet_Identity=True)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/SingletLKN.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [-1.856391565854298e-07+3.936838006505238e-11j, 4.967969922867994e-11+4.02260741273559e-11j,
                  4.073361587451076e-12-2.84491472224807e-12j, 3.320861996870726e-14+1.0772044017941923e-14j,
                  5.3586014368057094e-15-6.6442657672558726e-15j, 4.7342357348904615e-16+1.2237583111053118e-16j,
                  4.9950975578365465e-17-9.403776405933943e-18j]
correct_results = [-1 * correct_results[i] for i in range(len(correct_results))]
compare_c_results(res, correct_results, 1e-15)





#I = 0, I_3 = 0
O1 = twoHO(rep=(1/2,1/2), I=0, I3=0, A=KaonC, B=Nucleon)#( KaonC(-1/2) * Nucleon(1/2) - KaonC(1/2) * Nucleon(-1/2) )/(2**(1/2))
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Lambda())
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/SingletKCNL.hdf5')
#<Corr>{isosinglet_kbar_nucleon G1u_1 [P=(0,0,0) A1u SS_0] [P=(0,0,0) G1g SS_0]}{lambda P=(0,0,0) G1u_1 SS_0}</Corr>

hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'antikaon_ds',
          Momentum = (0,0,0), LGIrrep = 'A1u', Displacement = 'SS_0')

hadron2 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,1), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g', Displacement = 'SS_0', dlen='dlen0')

Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_baryon_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'G1u_1', Hadron1 = hadron1, Hadron2 = hadron2, OpNum = 0, strangeness = 1)

Hadron3 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'lambda_uds',
          Momentum = (0,0,0), LGIrrep = 'G1u_1', Displacement = 'SS_0', dlen='dlen0')


hadrons = [Hadron3, Two_Hadron_Sink]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128,
                                     Use_Triplet_Identity=True)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/SingletKCNL.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [-2.453046536078498e-07+1.0634277331752387e-10j, 6.55940959493042e-11+6.320193329386688e-11j,
      2.57574978700462e-12+1.458520635202231e-14j, 9.869719463404791e-14-6.320492716406146e-14j,
      3.1185324659684934e-15+3.9281264505992014e-15j, -4.623503477066217e-16-3.3071543492267715e-16j,
      -5.720905245139752e-17-3.387437491360553e-17j]

correct_results = [-1 * correct_results[i] for i in range(len(correct_results))]
compare_c_results(res, correct_results, 1e-15)



O1 = twoHO(rep=(1,1), I=1, I3=1, A=Pion, B=Pion)#(Pion(0) * Pion(1) - Pion(1) * Pion(0))/(2**(0.5))
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/Isotripletpppp.hdf5')

#{isotriplet_pion_pion T1up_3 [P=(0,0,1) A2m SS_1] [P=(0,0,-1) A2m SS_1]}
#{isotriplet_pion_pion T1up_3 [P=(0,0,1) A2m SS_1] [P=(0,0,-1) A2m SS_1]}</Corr>

hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isovector_du',
          Momentum = (0,0,1), LGIrrep = 'A2m', Displacement = 'SS_1')

hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = 'SS_1')

Two_Hadron_Sink = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'T1up_3', Hadron1 = hadron1, Hadron2 = hadron2, OpNum = 0, strangeness = 0)



hadron3 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isovector_du',
          Momentum = (0,0,1), LGIrrep = 'A2m', Displacement = 'SS_1')

hadron4 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = 'SS_1')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'T1up_3', Hadron1 = hadron3, Hadron2 = hadron4, OpNum = 0, strangeness = 0)

hadrons = [Two_Hadron_Sink, Two_Hadron_Source]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128,
                                     Use_Triplet_Identity=True)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/Isotripletpppp.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
correct_results = [0.5016497816120745-9.268956389357396e-11j,-6.469027564321331e-05-3.9866657853468165e-13j,
                 9.62763449787062e-07+3.5685344559645596e-15j, 1.5912905743209582e-07+3.9826124434250074e-15j,
                 1.029310126361129e-08+8.660584016016557e-16j, 1.9615588503976824e-09+2.813688785445631e-17j,
                 -1.3250172425844278e-10+3.9202474857141445e-16j, 1.0652951097991211e-10+2.1292838959178803e-16j]

compare_c_results(res, correct_results, 1e-15)

O1 = Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1,1), I=1, I3=1, A=Pion, B=Pion))#bar(Pion(1) * Pion(0) - Pion(0) * Pion(1)) / (2**0.5)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/P_IsoTripletPP.hdf5')


#<Corr>{pion P=(0,0,0) T1up_3 SS_0} {isotriplet_pion_pion T1up_3 [P=(0,0,1) A2m SS_1] [P=(0,0,-1) A2m SS_1]}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'T1up_3', Displacement = 'SS_0')

hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isovector_du',
          Momentum = (0,0,1), LGIrrep = 'A2m', Displacement = 'SS_1')
hadron3 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = 'SS_1')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'T1up_3', Hadron1 = hadron2, Hadron2 = hadron3, OpNum = 0, strangeness = 0)
hadrons = [Hadron1, Two_Hadron_Source]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/P_IsoTripletPP.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [-0.01705939621432921+2.6603397858967667e-12j, -0.0014696825931435242+2.0855574313806832e-13j,
                    -0.00012465409755524717+4.782099150045087e-13j, -1.0373391320468922e-05+8.273672765877032e-14j,
                    -4.5396166414579006e-07+7.547742714231752e-15j, -3.658885046922376e-08+7.732876916739392e-15j,
                    -4.755738142598925e-09+4.927950734076989e-15j, -5.830128321273885e-10+1.575373692106411e-15j]

compare_c_results(res, correct_results, 1e-15)


O1 = Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1,1), I=1, I3=1, A=Pion, B=Pion))#bar(Pion(0) * Pion(1) - Pion(1) * Pion(0)) / (2**0.5)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/P_IsoTripletPP.hdf5')


#<Corr>{pion P=(0,0,0) T1up_3 SS_0} {isotriplet_pion_pion T1up_3 [P=(0,0,1) A2m SS_1] [P=(0,0,-1) A2m SS_1]}</Corr>
#<Corr>{pion P=(0,0,0) T1up_3 SS_0} {isotriplet_pion_pion T1up_3 [P=(0,0,1) A2m SS_1] [P=(0,0,-1) A2m SS_1]}</Corr>

Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'T1up_3', Displacement = 'SS_0')

hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isovector_du',
          Momentum = (0,0,1), LGIrrep = 'A2m', Displacement = 'SS_1')
hadron3 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,1), Flavor = 'isovector_du',
          Momentum = (0,0,-1), LGIrrep = 'A2m', Displacement = 'SS_1')

Two_Hadron_Source = TwoHadron(File_Info_Path = '../Hadron_Info/meson_meson_operators.h5',
          Total_Momentum = (0,0,0), LGIrrep = 'T1up_3', Hadron1 = hadron2, Hadron2 = hadron3, OpNum = 0, strangeness = 0)
hadrons = [Hadron1, Two_Hadron_Source]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)

perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(8):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/P_IsoTripletPP.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [0.01705939621432921-2.6603397842104298e-12j, 0.001469682593143524-2.08555790740222e-13j, 0.00012465409755524722-4.782099199240265e-13j,
                  1.037339132046892e-05-8.273672872987712e-14j, 4.539616641457901e-07-7.547742713575824e-15j, 3.658885046922377e-08-7.732876921753064e-15j,
                  4.7557381425989245e-09-4.927950735107986e-15j, 5.830128321273886e-10-1.5753736923492618e-15j]

correct_results = [-1 * i for i in correct_results]
compare_c_results(res, correct_results, 1e-15)


O1 = Kaon(1/2)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/K+K+B.hdf5')


# <Corr>{kaon P=(0,0,0) A1u_1 SS_0} {kaon P=(0,0,0) A1u_1 SS_0}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u_1', Displacement = 'SS_0')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u_1', Displacement = 'SS_0')
hadrons = [Hadron1, Hadron2]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(5):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/K+K+B.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)


correct_results = [5.313235228930697-1.2374852500411735e-10j, 0.3133572739538783-4.988096974483457e-05j, 0.03609650510588132+6.839565236897468e-06j,
                  0.004810827480657285-6.399942130230121e-07j,0.0006508455450624747-3.150678487208817e-06j]
compare_c_results(res, correct_results, 1e-15)


# <Corr>{kaon P=(0,0,1) A2_1 SS_1}  {kaon P=(0,0,1) A2_1 SS_1}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,1), LGIrrep = 'A2_1', Displacement = 'SS_1')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,1), LGIrrep = 'A2_1', Displacement = 'SS_1')
hadrons = [Hadron1, Hadron2]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/K+K+B.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [0.4917000173547394-1.5147355808889373e-11j, 0.010757429734652414+0.002969804584046444j, 0.0007009890811380045+0.0001437055144395958j,
                  0.00012234691581427997+0.0001378371121389662j, 1.0880171685894247e-05+1.1880906853283298e-05j,
                   2.554982966206718e-06+7.397305852454038e-07j,3.548861749885612e-08+2.028180053069103e-07j]
compare_c_results(res, correct_results, 1e-15)



# <Corr>{kaon P=(0,0,0) A1u_1 SS_0} {kaon P=(0,0,0) A1u_1 SD_1}</Corr>:
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u_1', Displacement = 'SS_0')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u_1', Displacement = 'SD_1')
hadrons = [Hadron1, Hadron2]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/K+K+B.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [-5.485080478789817e-06-0.01773871949561849j, 1.2004716972565874e-05-0.0010292024430821243j,
                  -4.9036749771123316e-05-0.00012062606532584394j, 1.8042991178069682e-06-1.2415030815877924e-05j,
                  -1.647752403418175e-06-1.5108203736068658e-06j, -2.533833425994433e-07-2.4602977091122804e-07j,
                  7.3440529730488375e-09-3.5151051239063066e-08j]
compare_c_results(res, correct_results, 1e-15)



# <Corr>{kaon P=(0,0,0) A1u_1 SD_1} {kaon P=(0,0,0) A1u_1 SD_1}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u_1', Displacement = 'SD_1')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u_1', Displacement = 'SD_1')
hadrons = [Hadron1, Hadron2]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/K+K+B.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [-0.17995320205903043+4.301835363623718e-12j,4.736206549745513e-05+1.0788939367028247e-06j,
                   4.916165496289272e-06-3.212047527632545e-07j,5.800200199421669e-07+5.987670769104738e-08j,
                   3.41753249823348e-08-5.698025704292855e-09j, 2.158584017211272e-08-3.208260915533398e-09j,
                   7.348582635936443e-09+2.435616111205428e-09j]
compare_c_results(res, correct_results, 1e-15)


#<Corr>{kaon P=(0,0,0) A1u_1 SD_1} {kaon P=(0,0,0) A1u_1 SS_0}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u_1', Displacement = 'SD_1')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'kaon_su',
          Momentum = (0,0,0), LGIrrep = 'A1u_1', Displacement = 'SS_0')
hadrons = [Hadron1, Hadron2]


complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/K+K+B.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [-5.48508341652711e-06+0.01773871949443836j, 0.000255975402372115+0.0007132377278121097j,
                  1.4913873046250119e-05+0.00011228909346855395j, -1.5952375927729903e-05+1.9478611410040054e-05j,
                  -1.3983790034077655e-06+2.446921196986814e-06j, 3.1604899264361626e-07+3.8513400770803673e-07j,
                  -1.1400610785546163e-08+5.296440773137943e-08j]
compare_c_results(res, correct_results, 1e-15)



#<Corr>{nucleon P=(0,0,0) G1g_1 SS_0} {nucleon P=(0,0,0) G1g_1 SS_0}</Corr> 
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0', dlen='dlen0')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0', dlen='dlen0')
hadrons = [Hadron1, Hadron2]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/NN.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [0.0003014282195708719+1.4478640785371858e-07j, 4.284943031770383e-06+1.5498840936793988e-06j,
                  1.4422620984853958e-07-4.5033456557986815e-08j, 2.6513636626996723e-09-2.7247473235876045e-09j,
                  4.846041404226831e-11-2.9327588702431805e-11j, 1.5809896198140648e-12-1.6502696227215908e-12j,
                   -1.3871318870420694e-14+2.122202737228897e-14j]

compare_c_results(res, correct_results, 1e-15)


#<Corr>{nucleon P=(0,0,0) G1g_2 SS_0} {nucleon P=(0,0,0) G1g_2 SS_0}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_2', Displacement = 'SS_0', dlen='dlen0')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_2', Displacement = 'SS_0', dlen='dlen0')
hadrons = [Hadron1, Hadron2]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/NN.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
correct_results = [0.00030039787195030546-1.744900320148103e-07j, 4.230408254585984e-06+1.509849509210354e-06j,
                  1.402111854452368e-07-4.7661673187297724e-08j, 2.525481797910427e-09-2.6385986341764356e-09j,
                  3.7646453223177984e-11-2.8278297060512957e-11j, 1.6563771294195345e-12-1.4170101385674218e-12j,
                  -2.0993899502750057e-14+1.9596491324918984e-14j]

compare_c_results(res, correct_results, 1e-15)



O1 = Lambda(0)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/L0L0.hdf5')

#<Corr>{lambda P=(0,0,1) G2_1 SS_0} {lambda P=(0,0,1) G2_1 SS_0}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'lambda_uds',
          Momentum = (0,0,1), LGIrrep = 'G2_1', Displacement = 'SS_0', dlen='dlen0')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'lambda_uds',
          Momentum = (0,0,1), LGIrrep = 'G2_1', Displacement = 'SS_0', dlen='dlen0')
hadrons = [Hadron1, Hadron2]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res = []
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/L0L0.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)

correct_results = [-3.077753186553501e-05+2.7807329548041602e-09j, 9.12736477548813e-09-4.1764274918236207e-10j,
                  2.7160280351221543e-10-3.238324688617527e-10j, 6.876717886410605e-12-8.175758060480078e-12j,
                  -2.939490924514951e-14+6.506758142108084e-14j, -7.518908585422305e-15-5.584906587482788e-15j,
                  5.136559547125595e-16+3.3689353299549526e-16j]
compare_c_results(res, correct_results, 1e-15)

#<Corr>{lambda P=(0,0,1) G2_1 SD_12} {lambda P=(0,0,1) G2_1 SS_0}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'lambda_uds',
          Momentum = (0,0,1), LGIrrep = 'G2_1', Displacement = 'SD_12', dlen='dlen1')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'lambda_uds',
          Momentum = (0,0,1), LGIrrep = 'G2_1', Displacement = 'SS_0', dlen='dlen0')
hadrons = [Hadron1, Hadron2]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128,
                                     Use_Triplet_Identity=True)
res = []
import time
start = time.time()
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/L0L0.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
end = time.time()
print(f'Required_Time = {100 * (end - start)} ms')

correct_results = [1.4533169771377099e-08+1.9846226309445597e-07j, -4.489387156951679e-10+6.686084334939451e-09j,
                  9.687107204435481e-12+1.0437245470148878e-10j, 6.329673156004688e-12-1.655022415807599e-12j,
                  1.786266652335234e-13-1.8146430381357077e-14j, 7.411371889809892e-15-1.409273270688939e-14j,
                  3.3472055797573277e-16-3.448345273168473e-16j]


compare_c_results(res, correct_results, 1e-15)



#<Corr>{lambda P=(0,0,1) G2_1 SD_12} {lambda P=(0,0,1) G2_1 SD_12}</Corr>
Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'lambda_uds',
          Momentum = (0,0,1), LGIrrep = 'G2_1', Displacement = 'SD_12', dlen='dlen1')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'lambda_uds',
          Momentum = (0,0,1), LGIrrep = 'G2_1', Displacement = 'SD_12', dlen='dlen1')
hadrons = [Hadron1, Hadron2]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulators_strange = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_s_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': perambulators_strange, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128,
                                     Use_Triplet_Identity=True)
res = []
import time
start = time.time()
for t in range(7):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/L0L0.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res.append(test0_contracted)
end = time.time()
print(f'Required Time = {100 * (end - start)} ms')

correct_results = [1.1250771842230787e-05-1.2725451539131117e-09j, 1.9997680734186264e-07+3.364074983744315e-08j,
                  7.624568099318903e-10+4.292937901592418e-10j, -1.466024606710914e-11-5.607324966878773e-11j,
                  8.706009241463271e-13-1.7835767907929031e-12j, -1.4792132213798587e-13-2.88898170068082e-13j,
                  -4.2507422263273726e-15-5.184910676747679e-15j]



compare_c_results(res, correct_results, 1e-15)


O1 = Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/Pi+Pi+.hdf5')

Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'A1um_1', Displacement = 'SS_0')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'A1um_1', Displacement = 'SS_0')
hadrons = [Hadron1, Hadron2]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': None, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res1 = []
res2 = []
for t in range(9):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/Pi+Pi+.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = None)
    res1.append(test0_contracted)

pion = [0 for i in range(9)]
pion[0] = (4.280143667070952-1.6350227456098284e-10j)
pion[1] = (0.19890361370339574+2.0411337069269912e-19j)
pion[2] = (0.018045616035938555-9.287810014024311e-21j)
pion[3] = (0.0018946498116805485-1.1890622859063506e-20j)
pion[4] = (0.00020157319714492162-1.1739847651772713e-24j)
pion[5] = (1.8042330243095597e-05-6.394253214143184e-23j)
pion[6] = (2.2887691463923263e-06-3.339507463452103e-24j)
pion[7] = (3.0520964074138293e-07-4.321697482287652e-24j)
pion[8] = (1.3201802790095467e-07-2.0338252452333878e-25j)

for i in range(len(res1)):
    print(f'pion({i}) = {res1[i]}')
    print(f'diff    = {res1[i]-pion[i]}')
    print('______')

for i in range(len(res1)):
    tst = res1[i]-pion[i]
    if np.abs(tst.imag) >= 1e-8 or np.abs(tst.real) >= 1e-8:
        print(False)
        raise ValueError('Failed to reproduce correct results')
    else:
        print(True)

Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isovector_du',
          Momentum = (0,0,1), LGIrrep = 'A2m_1', Displacement = 'SS_1')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isovector_du',
          Momentum = (0,0,1), LGIrrep = 'A2m_1', Displacement = 'SS_1')
hadrons = [Hadron1, Hadron2]
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': None, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
res1 = []
res2 = []
for t in range(9):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/Pi+Pi+.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = None)
    res1.append(test0_contracted)
pion = [0 for i in range(9)]
pion[0] = (0.39681125942845186-2.4863527388211196e-11j)
pion[1] = (0.006854588941944439+0.0018739050089830213j)
pion[2] = (0.00035147845393663704+7.051514602640161e-05j)
pion[3] = (4.853857783263503e-05+5.5342007300572974e-05j)
pion[4] = (3.512518353425199e-06+3.784624963843414e-06j)
pion[5] = (6.305396259865426e-07+1.6889171994483086e-07j)
pion[6] = (6.545752107319039e-09+4.143195530282928e-08j)
pion[7] = (6.449180817163239e-10+2.1060350468251335e-09j)
pion[8] = (1.2538175809018842e-09-1.1807728582266968e-10j)

for i in range(len(res1)):
    print(f'pion({i}) = {res1[i]}')
    print(f'diff    = {res1[i]-pion[i]}')
    print('______')


for i in range(len(res1)):
    tst = res1[i]-pion[i]
    if np.abs(tst.imag) > 1e-8 or np.abs(tst.real) > 1e-8:
        print(False)
        print('tst: ', tst)
        raise ValueError('Failed to reproduce correct results')
    else:
        print(True)


O1 = sigma(0)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/SigmaSigmaB.hdf5')


Hadron1 = Hadron(File_Info_Path='../Hadron_Info/meson_operators.h5', Hadron_Type='meson_operators', Hadron_Position=(1, 0), 
                 Flavor='isoscalar', Momentum=[0, 0, 0], LGIrrep='A1gp_1', Displacement='SS_0')
Hadron2 = Hadron(File_Info_Path='../Hadron_Info/meson_operators.h5', Hadron_Type='meson_operators', Hadron_Position=(0, 0),
                 Flavor='isoscalar', Momentum=[0, 0, 0], LGIrrep='A1gp_1', Displacement='SS_0')

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = ['../Data/M_Tests/light_quark_perambulator_quda_5_More.hdf5', 
                                                              '../Data/M_Tests/light_quark_perambulator_quda_5.hdf5'], 
                                         Device = device, Double_Reading = True, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': None, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/M_Tests/mode_doublets_5.hdf5', Device = device, cplx128 = complex128)

Contractor = PyCorrTorch(SinkTime=4, SourceTime=0, Hadrons=[Hadron1, Hadron2], Path_Wicktract='../WickDiagrams/SigmaSigmaB.hdf5')
Contracted = Contractor.TorchTractor(All_Perambulators=perambulator, ModeDoublets=modeDoublet)
tst = Contracted-(39108.66379277315-5.5155943025067997e-11j)
if tst.imag > 1e-12 or tst.real > 1e-12:
    raise ValueError('Failed to reproduce M-Results')


Hadron3 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (0,0), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'A1um_1', Displacement = 'SS_0')
Hadron4 = Hadron(File_Info_Path = '../Hadron_Info/meson_operators.h5', Hadron_Type = 'meson_operators', Hadron_Position = (1,0), Flavor = 'isovector_du',
          Momentum = (0,0,0), LGIrrep = 'A1um_1', Displacement = 'SS_0')
hadrons = [Hadron3, Hadron4]

complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = ['../Data/M_Tests/light_quark_perambulator_quda_5.hdf5'], 
                                         Device = device, Double_Reading = True, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': None, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/M_Tests/mode_doublets_5.hdf5', Device = device, cplx128 = complex128)
test0            = PyCorrTorch(SinkTime = 4, SourceTime = 0, 
                               Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/Pi+Pi+.hdf5')
test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = None)
tst = test0_contracted-(78.13719804676474-5.081876067246993e-16j)
if tst.imag > 1e-12 or tst.real > 1e-12:
    raise ValueError('Failed to reproduce M-Results')


O1 = Nucleon(1/2)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = '../WickDiagrams/NN.hdf5')
complex128          = True
device              = get_best_device(use_gpu = True, cplx128 = complex128)
perambulators_light = PyTor_Perambulator(Path_Perambulator = '../Data/perambs_ud_1.hdf5', 
                                         Device = device, Double_Reading = False, cplx128 = complex128)
perambulator        = {'Light': perambulators_light, 'Strange': None, 'Charm': None}
modeDoublet         = PyTor_MDoublet(Path_ModeDoublet = '../Data/mode_doublets_1.hdf5', Device = device, cplx128 = complex128)
modeTriplet_old         = PyTor_MTriplet(Path_ModeTriplet = '../Data/mode_triplets_1_N.hdf5', Device = device, cplx128 = complex128)
modeTriplet = {}
for i in range(9):
    new_name = 'px0_py0_pz0_ddir0_t'+str(i)
    modeTriplet[new_name] = modeTriplet_old[f'px0_py0_pz0_ddir0_dlen0_t{str(i)}']



Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_1', Displacement = 'SS_0')
hadrons = [Hadron1, Hadron2]
res1 = []
for t in range(9):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/NN.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res1.append(test0_contracted)









Hadron1 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (1,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_2', Displacement = 'SS_0')
Hadron2 = Hadron(File_Info_Path = '../Hadron_Info/baryon_operators.h5', Hadron_Type = 'baryon_operators', Hadron_Position = (0,0), Flavor = 'nucleon_uud',
          Momentum = (0,0,0), LGIrrep = 'G1g_2', Displacement = 'SS_0')
hadrons = [Hadron1, Hadron2]
res2 = []
for t in range(9):
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = '../WickDiagrams/NN.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = modeTriplet)
    res2.append(test0_contracted)




final_res = []

if len(res1) != len(res2):
    raise TypeError('Error!')
else:
    for i in range(len(res1)):
        res_i = (res1[i] + res2[i]) / 2
        final_res.append(res_i)
nucleon = [0 for i in range(9)]
for t in range(9):
    P = perambulators_light[f'srcTime0_snkTime{t}']
    Bso = modeTriplet['px0_py0_pz0_ddir0_t0']
    Bsi = modeTriplet[f'px0_py0_pz0_ddir0_t{t}']
    CN1 = torch.einsum('klm,KLM,kK,lL,mM->', Bsi, Bso.conj(),P[0,0],P[0,0], P[1,1]) - torch.einsum('klm,KLM,kK,lL,mM->', Bsi, Bso.conj(),P[0,0],P[0,1], P[1,0])
    CN2 = torch.einsum('klm,KLM,kK,lL,mM->', Bsi, Bso.conj(),P[1,1],P[0,0], P[1,1]) - torch.einsum('klm,KLM,kK,lL,mM->', Bsi, Bso.conj(),P[1,1],P[0,1], P[1,0])
    to_add = 3 * (CN1.item()+CN2.item())/2
    nucleon[t] = to_add

for i in range(len(final_res)):
    print(f'nucleon({i}) = {final_res[i].real }')
    print(f'diff    = {final_res[i].real-nucleon[i].real}')
    print('______')


for i in range(len(res1)):
    tst = final_res[i]-nucleon[i]
    if np.abs(tst.imag) > 1e-15 or np.abs(tst.real) > 1e-15:
        print(False)
        raise ValueError('Failed to reproduce Colins Results')
    else:
        print(True)
