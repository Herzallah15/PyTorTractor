from Hadron_Info_Converter import *

class TwoHadron:
    def __init__(self, File_Info_Path = None, Total_Momentum = None, LGIrrep = None, Hadron1 = None, Hadron2 = None, OpNum = None):
        self.File_Info_Path = File_Info_Path
        self.Total_Momentum = Total_Momentum
        self.LGIrrep        = LGIrrep
        self.Hadron1        = Hadron1
        self.Hadron2        = Hadron2
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
        if twohadron_type in ['MesonBaryon', 'MesonMeson']:
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
                H1_Group = self.Hadron1.getGroup() + str(self.Hadron_TotalCombi[i][3])
                
                H2_Momentum = tuple(self.Hadron_TotalCombi[i][4:7].tolist())
                H2_Group = self.Hadron2.getGroup() + str(self.Hadron_TotalCombi[i][7])
                
                hdrn1 = Hadron(File_Info_Path = self.Hadron1.getFile_Info_Path(), Hadron_Type = self.Hadron1.getHadron_Type(),
                               Hadron_Position = self.Hadron1.getHadron_Position(), Flavor = self.Hadron1.getFlavor(),
                               Momentum = H1_Momentum, LGIrrep = H1_Group, 
                               Displacement = self.Hadron1.getDisplacement(), dlen = self.Hadron1.getDlen())
                hdrn2 = Hadron(File_Info_Path = self.Hadron2.getFile_Info_Path(), Hadron_Type = self.Hadron2.getHadron_Type(),
                               Hadron_Position = self.Hadron2.getHadron_Position(), Flavor = self.Hadron2.getFlavor(),
                               Momentum = H1_Momentum, LGIrrep = H1_Group, 
                               Displacement = self.Hadron2.getDisplacement(), dlen = self.Hadron2.getDlen())
                ForFactor = self.Numerical_Coefficients[i]
                T[f'combi_{i}'] = {'Hadrons': [hdrn1, hdrn2], 'Factor': ForFactor}
            self.alIn = T
        elif twohadron_type == 'MesonMeson':
            raise ValueError('We still cannot handle MesonMeson case')
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