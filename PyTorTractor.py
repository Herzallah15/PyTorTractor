from PyCorrTorch_SingleCor import *
from MH_PyTorTractor import *

class PyCorrTorch():
    def __init__(self, SinkTime = None, SourceTime = None, 
                 Hadrons = None, Path_Wicktract = None):
        self.SinkTime          = SinkTime
        self.SourceTime        = SourceTime
        self.Path_Wicktract    = Path_Wicktract

        Decomposed_Hadrons = Hadrons[0]
        for hdrn in self.Hadrons[1:]:
            Decomposed_Hadrons *= hdrn
        self.Hadrons_Map = Decomposed_Hadrons
        '''
        self_Hadrons_Map is of the form { 
          combi_0 : {'Hadrons': hdrn123, 'Factor': ForFactor},
          combi_1 : {'Hadrons': hdrn123, 'Factor': ForFactor},
          ...
            }
        '''
    def TorchTractor(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None):
        final_result = 0.0
        for i, combi in enumerate(self.Hadrons_Map):
            print('______')
            hadrons    = self.Hadrons_Map[combi]['Hadrons']
            num_Factor = self.Hadrons_Map[combi]['ForFactor']
            do_contration0 = PyCorrTorch_SingleCor(SinkTime = self.SinkTime, SourceTime = self.SourceTime, 
                                                   Hadrons = hadrons, Path_Wicktract = self.Path_Wicktract)
            do_contration1 = do_contration0.TorchTractor_SingleCor(All_Perambulators = All_Perambulators, 
                                                                  ModeDoublets = ModeDoublets, ModeTriplets = ModeTriplets)
            print(do_contration1)
            do_contration2 = combine_all(do_contration1)
            res = do_contration2 * num_Factor
            print(f'Correlator_{i} = {res}')
            final_result += res
        return final_result
        
        '''
    test0 = PyCorrTorch(SinkTime = t, SourceTime = 0, Hadrons = hadrons, Path_Wicktract = 'PyTor_Test_1P.hdf5')
    test0_contracted = test0.TorchTractor(All_Perambulators = perambulator, ModeDoublets = modeDoublet, ModeTriplets = None)
    res1.append(combine_all(test0_contracted))
        '''
