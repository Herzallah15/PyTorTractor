import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Source")))
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
from contractions_handler import *
from Hadron_Info_Converter import *
from Hadrontractions_Converter import *
from PyTorDefinitions import *
from PyTor_S_Definitions import *
from PyTorTractor_SingleHadron import *
from PyTorTractor import *
import time
from datetime import datetime
day = datetime.now().strftime("%Y%m%d")
start = time.time()
'''
Run to obtain the correlation matrix for the following set:
## $rho$ frame $P = [0, 0, 0]$ $T_{1u,1}^+$

- {rho P=[0, 0, 0] T1up_1 SS_0} = O1_1
- {rho P=[0, 0, 0] T1up_1 SS_1} = O1_2
- {isotriplet_pion_pion T1up_1 P=[0, 0, 0] S=0_+00_A2m_-00_A2m_0 A2m SS_1 A2m SS_1} = O2_1
- {isotriplet_pion_pion T1up_1 P=[0, 0, 0] S=0_+0+_A2m_-0-_A2m_0 A2m SS_0 A2m SS_0} = O2_2
- {isotriplet_kaon_kaonC T1up_1 P=[0, 0, 0] S=0_+00_A2_-00_A2_0 A2 SS_1 A2 SS_1} = O3

This results in:


{O1_1} {O1_1}
{O1_1} {O1_1}
{O1_1} {O2_1}
{O1_1} {O2_2}
{O1_1} {O3}

###

{O1_1} {O1_1}
{O1_1} {O1_1}
{O1_1} {O2_1}
{O1_1} {O2_2}
{O1_1} {O3}


###

{O2_1} {O1_1}
{O2_1} {O1_1}
{O2_1} {O2_1}
{O2_1} {O2_2}
{O2_1} {O3}


###

{O2_2} {O1_1}
{O2_2} {O1_1}
{O2_2} {O2_1}
{O2_2} {O2_2}
{O2_2} {O3}



###

{O3} {O1_1}
{O3} {O1_1}
{O3} {O2_1}
{O3} {O2_2}
{O3} {O3}


# Generating all Wick's Diagrams:
Cor_List = [Pion(1), twoHO(rep=(1,1), I=1, I3=1, A=Pion, B=Pion), twoHO(rep=(1/2,1/2), I=1, I3=1, A=Kaon, B=KaonC)]
for i, sink_op in enumerate(Cor_List):
    for j, source_op in enumerate(Cor_List):
        O1_ontime = OpTimeSlice(1, sink_op)
        O2_ontime = OpTimeSlice(0, bar(source_op))
        Correlator = O1_ontime * O2_ontime
        Result = Correlator.Laudtracto()
        wick_path = f'HerzDiagrams/{i}_{j}.hdf5'
        writeresults(Result, O1_ontime, O2_ontime, path = wick_path)

'''




print('Torch installed:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())


cnfigs = list(range(4, 185, 10))
source_times = range(127)
sink_times = range(127)

# Get config from command line argument
if len(sys.argv) > 1:
    cnf = int(sys.argv[1])
else:
    raise ValueError('You need to provide the configuer number')


triplets_path = lambda config: f'../from_elysium/mode_triplets_{config}.hdf5'
doublet_path = lambda config: f'/capstor/scratch/cscs/msalg/converted_doublets/X252/mode_doublets_{config}.hdf5'
perambulator_path = lambda config: f'/capstor/scratch/cscs/msalg/converted_perambs/X252/light_quark_perambulator_quda_{config}.hdf5'
b_path = 'Hadron_Info/baryon_operators.h5'
m_path = 'Hadron_Info/meson_operators.h5'
bOp = 'baryon_operators'
mOp = 'meson_operators'
twomOp = 'Hadron_Info/meson_meson_operators.h5'








pIon = lambda position, momentum, group, dis: Hadron(File_Info_Path = m_path, Hadron_Type = mOp, Hadron_Position = position, Flavor = 'isovector_du',
                                                     Momentum = momentum, LGIrrep = group, Displacement = dis)

kAon = lambda position, momentum, group, dis: Hadron(File_Info_Path = m_path, Hadron_Type = mOp, Hadron_Position = position, Flavor = 'kaon_su',
                                                     Momentum = momentum, LGIrrep = group, Displacement = dis)

kAonC = lambda position, momentum, group, dis: Hadron(File_Info_Path = m_path, Hadron_Type = mOp, Hadron_Position = position, Flavor = 'antikaon_ds',
                                                     Momentum = momentum, LGIrrep = group, Displacement = dis)


#{rho P=[0, 0, 0] T1up_1 SS_0} = O1_1
O1_1 = lambda position: pIon(position, (0,0,0), 'T1up_1', 'SS_0')

#{rho P=[0, 0, 0] T1up_1 SS_1} = O1_1
O1_2 = lambda position: pIon(position, (0,0,0), 'T1up_1', 'SS_1')


#{isotriplet_pion_pion T1up_1 P=[0, 0, 0] S=0_+00_A2m_ -00_A2m_0 A2m SS_1 A2m SS_1} = O2_1
O2_1 = lambda position1, position2: TwoHadron(File_Info_Path = twomOp, Total_Momentum = (0,0,0), LGIrrep = 'T1up_1',
               Hadron1 = pIon(position1, (1, 0, 0), 'A2m', 'SS_1'),
               Hadron2 = pIon(position2, (-1, 0, 0), 'A2m', 'SS_1'), OpNum = 0, strangeness = 0)

#{isotriplet_pion_pion T1up_1 P=[0, 0, 0] S=0_+0+_A2m_-0-_A2m_0 A2m SS_0 A2m SS_0} = O2_2
O2_2 = lambda position1, position2: TwoHadron(File_Info_Path = twomOp, Total_Momentum = (0,0,0), LGIrrep = 'T1up_1',
               Hadron1 = pIon(position1, (1, 0, 1), 'A2m', 'SS_0'),
               Hadron2 = pIon(position2, (-1, 0, -1), 'A2m', 'SS_0'), OpNum = 0, strangeness = 0)
#{isotriplet_kaon_kaonC T1up_1 P=[0, 0, 0] S=0_+00_A2_-00_A2_0 A2 SS_1 A2 SS_1}
O4 = lambda position1, position2: TwoHadron(File_Info_Path = twomOp, Total_Momentum = (0,0,0), LGIrrep = 'T1up_1',
               Hadron1 = kAon(position1, (1, 0, 0), 'A2', 'SS_1'),
               Hadron2 = kAonC(position2, (-1, 0, 0), 'A2', 'SS_1'), OpNum = 0, strangeness = 0)

all_in_one = [O1_1, O1_2, O2_1, O2_2, O4]

way = {0: 'rho_SS_0',
      1: 'rho_SS_1',
      2: 'isotriplet_pion_pion_T1up_1_SS=0_+00_A2m_-00_A2m_0_A2m SS_1 A2m SS_1',
      3: 'isotriplet_pion_pion_T1up_1_S=0_+0+_A2m_-0-_A2m_0_A2m_SS_0_A2m_SS_0',
      4: 'isotriplet_kaon_kaonC_T1up_1_S=0_+00_A2_-00_A2_0_A2_SS_1_A2_SS_1'}
wickM = {0: 0, 1: 0,
         2: 1, 3:1,
         4: 2}
involved_momenta = [(1,0,1), (-1,0,-1), (0,0,0), (1,0,0), (-1,0,0)]
ddir_max = 2
DPths = lambda x, y, z, ddir, t: f'px{x}_py{y}_pz{z}_ddir{ddir}_t{t}'
def MDths(t):
    path1 = [DPths(1,0,0, 0, t), DPths(-1,0,0, 0, t), DPths(1,0,1, 0, t), DPths(1,0,-1, 0, t)]
    for ddir in range(ddir_max):
        path1 += [DPths(1,0,1, ddir, t)]+ [DPths(-1,0,-1, ddir, t)] + [DPths(1,0,0, ddir, t)]+ [DPths(-1,0,0, ddir, t)]
        path1 += [DPths(1,0,1, -ddir, t)]+ [DPths(-1,0,-1, -ddir, t)] + [DPths(1,0,0, -ddir, t)]+ [DPths(-1,0,0, -ddir, t)]
    return path1
saving_path = f'{cnf}_{cnf+5}_2P_2P'
f = h5py.File(f'Results/{saving_path}.hdf5', 'w')
for config in cnfigs[cnf: cnf+5]:
    for srctime in source_times:
        res = {(0, 0): [], (0, 1): [], (0, 2): [], (0, 3): [], (0, 4): [],
               (1, 0): [], (1, 1): [], (1, 2): [], (1, 3): [], (1, 4): [],
               (2, 0): [], (2, 1): [], (2, 2): [], (2, 3): [], (2, 4): [],
               (3, 0): [], (3, 1): [], (3, 2): [], (3, 3): [], (3, 4): [],
              (4, 0): [], (4, 1): [], (4, 2): [], (4, 3): [], (4, 4): []} 
        for snktime in sink_times:
            P_paths = list(set([f'srcTime{snktime}_snkTime{srctime}', f'srcTime{srctime}_snkTime{snktime}', 
                                f'srcTime{srctime}_snkTime{srctime}', f'srcTime{snktime}_snkTime{snktime}']))
            M_paths = list(set(MDths(snktime) + MDths(srctime)))
            perambulator = {'Light': PyTor_Perambulator(Path_Perambulator = perambulator_path(config), Device = device,
                                                        Selected_Groups = P_paths, Double_Reading = False, cplx128 = complex128),
                            'Strange': None, 'Charm': None}
            modeDoublet  = PyTor_MDoublet(Path_ModeDoublet = doublet_path(config), Device = device, Selected_Groups = M_paths, cplx128 = complex128)

            for i_sink, sink_hadron in enumerate(all_in_one):
                for i_source, source_hadron in enumerate(all_in_one):
                    if i_sink <= 1:
                        Hadrons_sink = [sink_hadron((1,0))]
                    else:
                        Hadrons_sink = [sink_hadron((1,0), (1,1))]

                    if i_source <= 1:
                        Hadrons_source = [source_hadron((0,0))]
                    else:
                        Hadrons_source = [source_hadron((0,0), (0,1))]

                    Hadrons = Hadrons_sink + Hadrons_source
                    wick_path = f'HerzDiagrams/{wickM[i_sink]}_{wickM[i_source]}.hdf5'
                    sp0 = PyCorrTorch(SinkTime=snktime, current_time = None, SourceTime = srctime, Hadrons = Hadrons, Path_Wicktract = wick_path)
                    sp1 = sp0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = None, 
                                           all_SG_perambulators = None)
                    res[(i_sink, i_source)].append(sp1)
            del perambulator, modeDoublet
        grp = f.create_group(f'{config}/srcTime{srctime}')
        for (i, j), arrays in res.items():
            data = torch.stack(arrays, dim=0).cpu().numpy()
            grp.create_dataset(f'{way[i]}_{way[j]}', data=data)
f.close()
print('Task Done')
print('___________________')


end = time.time()
required_time = end - start
print(f'It took ', required_time, 'seconds')
