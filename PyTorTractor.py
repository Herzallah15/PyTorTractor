from PyTorDefinitions import *

class PyCorrTorch:
    def __init__(self, SinkTime = None, SourceTime = None, 
                 Hadrons = None, Path_Wicktract = None, 
                 #Path_Perambulator = None, Path_ModeDoublet = None, Path_ModeTriplet = None, useGPU = True, Device_ID = None
                ):
        #if None in (SinkTime, SourceTime, Hadrons, Path_Wicktract, Path_Perambulator):
        #    raise ValueError(error01)
        #if (Path_ModeDoublet is None) and (Path_ModeTriplet is None):
        #    raise ValueError(error02)
        self.SinkTime          = SinkTime
        self.SourceTime        = SourceTime
        self.Hadrons           = Hadrons
        self.Path_Wicktract    = Path_Wicktract


        # part which is commented out. Called PART_OUTCOMMENT_0
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
    def TorchTractor(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None):
        if (All_Perambulators is None):
            raise ValueError('The perambulators_dicts must be forwarded to TorchTractor as All_Perambulators = ...')
        if (ModeDoublets is None) and (ModeTriplets is None):
            er = 'TorchTractor must take as argument at least a ModeDoublet or a ModeTriplet.'
            raise ValueError(f'{er} one or both of the following arguments are missing: ModeDoublets = ..., ModeTriplets = ... ')

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
                        Mode_Tensors.append(ModeDoublets[path].conj())
                    elif self.hadron_type_mom_map[hadron]['T'] == 'B':
                        Mode_Tensors.append(ModeTriplets[path].conj())
                        Mode_Indices += index_map[hadron + (2,)]
                elif hadron[0] == 1:
                    if self.hadron_type_mom_map[hadron]['T'] == 'M':
                        Mode_Tensors.append(ModeDoublets[path])
                    elif self.hadron_type_mom_map[hadron]['T'] == 'B':
                        Mode_Tensors.append(ModeTriplets[path])
                        Mode_Indices += index_map[hadron + (2,)]
                Mode_Indices = Mode_Indices + ','
            return {'index': Mode_Indices, 'Tensor': Mode_Tensors}
        #comment_02
        def Perambulator_Setup(exp_prmp_container):
            Prmp_Indices = ''
            Prmp_Tensors = []
            for perambulator in exp_prmp_container:
                Prmp_Indices += index_map[perambulator.getQ()]
                Prmp_Indices += index_map[perambulator.getQ_Bar()]
                s             = perambulator.getS() - 1
                s_Bar         = perambulator.getS_Bar() - 1
                p_left        = perambulator.getH()[0]
                p_right       = perambulator.getH_Bar()[0]
                prmp_flavor   = perambulator.getFlavor()
                num_factor    = perambulator.getFF()
                if p_left   == 1 and p_right == 0:
                    time    = f'srcTime{self.SourceTime}_snkTime{self.SinkTime}'
                elif p_left == 0 and p_right == 1:
                    time    = f'srcTime{self.SinkTime}_snkTime{self.SourceTime}'
                elif p_left == 1 and p_right == 1:
                    time    = f'srcTime{self.SinkTime}_snkTime{self.SinkTime}'
                elif p_left == 0 and p_right == 0:
                    time    = f'srcTime{self.SourceTime}_snkTime{self.SourceTime}'
                else:
                    raise ValueError('Error in extracting perambulators from the Perambulator_Tensor_Dict')
                Prmp_Tensors.append((All_Perambulators[prmp_flavor][time][s, s_Bar, :, :] * num_factor))
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