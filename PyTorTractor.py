from PyTorDefinitions import *

class PyCorrTorch:
    def __init__(self, SinkTime = None, SourceTime = None, 
                 Hadrons = None, Path_Wicktract = None, 
                 Path_Perambulator = None, Path_ModeDoublet = None, Path_ModeTriplet = None, useGPU = True, Device_ID = None):
        if None in (SinkTime, SourceTime, Hadrons, Path_Wicktract, Path_Perambulator):
            raise ValueError(error01)
        if (Path_ModeDoublet is None) and (Path_ModeTriplet is None):
            raise ValueError(error02)
        self.SinkTime          = SinkTime
        self.SourceTime        = SourceTime
        self.Hadrons           = Hadrons
        self.Path_Wicktract    = Path_Wicktract
        self.Path_Perambulator = Path_Perambulator
        self.Path_ModeDoublet  = Path_ModeDoublet
        self.Path_ModeTriplet  = Path_ModeTriplet

        self.useGPU            = useGPU
        self.device            = get_best_device(use_gpu = self.useGPU, device_id = Device_ID, verbose = True)

        # Construct now the Perambulator_Super_Tensor
        with h5py.File(self.Path_Perambulator, 'r') as yunus:
            yunus1        = yunus[f'/PerambulatorData/srcTime{self.SourceTime}_snkTime{self.SinkTime}']
            N             = int(np.sqrt(yunus1['srcSpin1']['snkSpin1']['re'].shape[0]))
            P_SuperTensor = torch.zeros((4, 4, N, N), dtype=torch.complex128)
            for i in range(4):
                for j in range(4):
                    P_SuperTensor[i, j, :, :] = torch.complex(
                        torch.from_numpy(
                            yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['re'][:]).reshape(N, N), 
                        torch.from_numpy(
                            yunus1['srcSpin'+str(j+1)]['snkSpin'+str(i+1)]['im'][:]).reshape(N, N))
            #P^{s_{snk} s_{src} snkevn srcevn}
            g5                    = gamma(5, torch.complex128).to(self.device)
            g4                    = gamma(4, torch.complex128).to(self.device)
            gM                    = torch.matmul(g5, g4)
            self.P_SuperTensor    = P_SuperTensor.to(self.device)
            self.P_Re_SuperTensor = torch.einsum('ij,jnlm,nk->kiml', gM, self.P_SuperTensor, gM).conj()
            print(r'Perambulator_Tensor has been successfully constructed')

        # Construct now the ModeDoublet_Super_Tensor
        if self.Path_ModeDoublet is not None:
            with h5py.File(self.Path_ModeDoublet, 'r') as yunus:
                yunus1         = yunus['/ModeDoubletData']
                MD_SuperTensor = {}
                for group in yunus1:
                    MD_SuperTensor[group] = torch.complex(
                        torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N),
                        torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N)).to(dtype=torch.complex128).to(self.device)
                #G^{i j}
            self.MD_SuperTensor = MD_SuperTensor
            print(r'MD_Tensor has been successfully constructed')

        # Construct now the ModeTriplet_Super_Tensor
        if self.Path_ModeTriplet is not None:
            with h5py.File(self.Path_ModeTriplet, 'r') as yunus:
                yunus1         = yunus['/ModeTripletData']
                MT_SuperTensor = {}
                for group in yunus1:
                    MT_SuperTensor[group] = torch.complex(
                        torch.from_numpy(yunus1[group]['re'][:]).reshape(N,N,N),
                        torch.from_numpy(yunus1[group]['im'][:]).reshape(N,N,N)).to(dtype=torch.complex128).to(self.device)
                #G^{i j}
            self.MT_SuperTensor = MT_SuperTensor
            print(r'MT_Tensor has been successfully constructed')

        # Cluster the Diagrams
        self.clusters, self.WT_numerical_factors = cluster_extractor(Path_Diagrams = self.Path_Wicktract)

        # Sort the hadrons to baryons and mesons so later it becomes easier to extract MT and MD
        hadron_type_mom_map = {}
        for hdrn in self.Hadrons:
            hadron_type_mom_map[hdrn.getHadron_Position()] = {'T': hdrn_type(hdrn.getHadron_Type()), 'P': momentum(hdrn.getMomentum())}
        self.hadron_type_mom_map = hadron_type_mom_map
        



        # SpinStructure Combinations between the hadrons
        self.hadron_product = hadron_info_multiplier(*self.Hadrons)
        print('All combinations of hadron structures coefficients were generated')

        print('Insert now these combinations explicitly into the the clusters!')
        self.clusters_with_kies = [((outer_key, inner_key), 
                                    Final_Perambulator_Container(prpm_container, self.hadron_product).getExplicit_Perambulator_Containers() ) 
                               for outer_key, inner_dict in self.clusters.items() 
                               for inner_key, prpm_container in inner_dict.items()]

        print('Each cluster is now splitted into many clusters with various explicit spin combinations')
#commet_01
    def TorchTractor(self):
#comment_02
        def Modes_Setup(outer_cluster, exp_prmp_container):
            dis_paths = {}
            for prp in exp_prmp_container:
                if prp.getH() not in dis_paths:
                    dis_paths[prp.getH()] = ddir(prp.getDis())
                if prp.getH_Bar() not in dis_paths:
                    dis_paths[prp.getH_Bar()] = ddir(prp.getDis_Bar())
            Mode_Indices = ''
            Mode_Tensors = []
            for hadron in outer_cluster:
                Mode_Indices += index_map[hadron + (0,)]
                Mode_Indices += index_map[hadron + (1,)]
                mntm          = self.hadron_type_mom_map[hadron]['P']
                disp          = dis_paths[hadron]
                time          = 't'+str(self.SourceTime)
                path          = mntm + '_' + disp + '_' + time
                if hadron[0] == 0:
                    if self.hadron_type_mom_map[hadron]['T'] == 'M':
                        Mode_Tensors.append(self.MD_SuperTensor[path].conj())
                    elif self.hadron_type_mom_map[hadron]['T'] == 'B':
                        Mode_Tensors.append(self.MT_SuperTensor[path].conj())
                        Mode_Indices += index_map[hadron + (2,)]
                elif hadron[0] == 1:
                    if self.hadron_type_mom_map[hadron]['T'] == 'M':
                        Mode_Tensors.append(self.MD_SuperTensor[path])
                    elif self.hadron_type_mom_map[hadron]['T'] == 'B':
                        Mode_Tensors.append(self.MT_SuperTensor[path])
                        Mode_Indices += index_map[hadron + (2,)]
                Mode_Indices = Mode_Indices + ','
            return {'index': Mode_Indices, 'Tensor': Mode_Tensors}
#comment_03
        def Perambulator_Setup(exp_prmp_container):
            Prmp_Indices = ''
            Prmp_Tensors = []
            for perambulator in exp_prmp_container:
                Prmp_Indices += index_map[perambulator.getQ()]
                Prmp_Indices += index_map[perambulator.getQ_Bar()]
                s             = perambulator.getS() - 1
                s_Bar         = perambulator.getS_Bar() - 1
                if (perambulator.getH()[0] == 0) and (perambulator.getH_Bar()[0] == 1):
                    Prmp_Tensors.append(self.P_Re_SuperTensor[s, s_Bar, :, :])
                elif (perambulator.getH()[0] == 1) and (perambulator.getH_Bar()[0] == 0):
                    Prmp_Tensors.append( (self.P_SuperTensor[s, s_Bar, :, :] * perambulator.getFF()) )
                elif perambulator.getH()[0] == perambulator.getH_Bar()[0]:
                    Prmp_Tensors.append( (self.P_SuperTensor[s, s_Bar, :, :] * perambulator.getFF()) )
            return {'index': Prmp_Indices, 'Tensor': Prmp_Tensors}
        clusters_with_kies_copy = []
        for full_cluster in self.clusters_with_kies:
            modes_info    = Modes_Setup(full_cluster[0][0], full_cluster[1][0])
            modes_indices = modes_info['index'][:-1]
            modes_tensors = modes_info['Tensor']
            prmp_list     = []
            extractor     = Perambulator_Setup(full_cluster[1][0])
            prmp_indizes  = extractor['index']
            prmp_list.append(extractor['Tensor'])
            for prmp_container in full_cluster[1][1:]:
                peram_info = Perambulator_Setup(prmp_container)
                if peram_info['index'] != prmp_indizes:
                    raise ValueError('Something wrong with Perambulator_Extractor')
                prmp_list.append(peram_info['Tensor'])
            Perambulators = torch.stack([Tensor_Product(Qs) for Qs in prmp_list], dim=0)
            results      = torch.einsum(f'{modes_indices},Z{prmp_indizes}->Z', *modes_tensors, Perambulators)
            results      = torch.sum(results, dim=0)
            clusters_with_kies_copy.append((full_cluster[0], results))
        return clusters_with_kies_copy, self.WT_numerical_factors