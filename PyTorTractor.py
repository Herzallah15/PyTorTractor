from PyTorDefinitions import *
import os, psutil

class PyCorrTorch:
    def __init__(self, SinkTime = None, SourceTime = None, 
                 Hadrons = None, Path_Wicktract = None):
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
            hadron_type_mom_map[hdrn.getHadron_Position()] = {'T': hdrn_type(hdrn.getHadron_Type()), 'P': hdrn.getMomentum_Value()}
        self.hadron_type_mom_map = hadron_type_mom_map
        



        # SpinStructure Combinations between the hadrons
        self.hadron_product = hadron_info_multiplier(*self.Hadrons)
        print('All combinations of hadron structures coefficients were generated')
        print('Insert now these combinations explicitly into the the clusters!')
        self.clusters_with_kies = [((outer_key, inner_key), 
                                    [Final_Perambulator_Container(prpm_container, self.hadron_product).getModeInfos(),
                                    Final_Perambulator_Container(prpm_container, self.hadron_product).getExplicit_Perambulator_Containers()] ) 
                               for outer_key, inner_dict in self.clusters.items() 
                               for inner_key, prpm_container in inner_dict.items()]
        print('Each cluster is now splitted into many clusters with various explicit spin combinations')

    def TorchTractor(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None):
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024**2 
        print(f"Speicherverbrauch: {mem:.1f} MB")
        if (All_Perambulators is None):
            raise ValueError('The perambulators_dicts must be forwarded to TorchTractor as All_Perambulators = ...')
        if (ModeDoublets is None) and (ModeTriplets is None):
            er = 'TorchTractor must take as argument at least a ModeDoublet or a ModeTriplet.'
            raise ValueError(f'{er} One or both of the following arguments are missing: ModeDoublets = ..., ModeTriplets = ... ')
        print(f'{len(self.clusters_with_kies)} tensor contractions to be performed')
        cntrctns_cntr = 0
        def Perambulator_Laph(exp_prmp_container):
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / 1024**2 
            print(f"Speicherverbrauch Anfang von Perambulator_Laph: {mem:.1f} MB")
            Prmp_Indices_In  = ''
            Prmp_Indices_Out = ''
            Prmp_Tensors = []
            for perambulator in exp_prmp_container:
                Q_Info, Q_Bar_Info = perambulator.getQ(), perambulator.getQ_Bar()
                Prmp_Indices_In   += spin_index_map[Q_Info] + spin_index_map[Q_Bar_Info]
                Prmp_Indices_In   += index_map[Q_Info] + index_map[Q_Bar_Info] + ','
                Prmp_Indices_Out  += spin_index_map[Q_Info] + spin_index_map[Q_Bar_Info]
                p_left, p_right   = perambulator.getH()[0], perambulator.getH_Bar()[0]
                prmp_flavor       = perambulator.getFlavor()
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
                Prmp_Tensors.append(All_Perambulators[prmp_flavor][time])
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / 1024**2
                print(f"Speicherverbrauch_End von Perabmulaotr_Laph: {mem:.1f} MB")
            return {'index_In': Prmp_Indices_In[:-1], 'index_Out': Prmp_Indices_Out, 'Tensor': Prmp_Tensors}
        clusters_with_kies_copy = []
        for full_cluster in self.clusters_with_kies:
            Doublet_Triplet_Tensor_Info = full_cluster[1][0]['MDT_Info']
            DT_Index  = ','.join(full_cluster[1][0]['Mode_Index_Info'])
            if isinstance(Doublet_Triplet_Tensor_Info, tuple):
                print('One ModeDoublet/Triplet for a spin combination')
                M_Tensors = []
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / 1024**2 
                print(f"Speicherverbrauch Vor M_Tensors: {mem:.1f} MB")
                for path in Doublet_Triplet_Tensor_Info:
                    if path[0] == '0':
                        final_path = path[3:]+'_t'+str(self.SourceTime)
                        if path[1] == 'D':
                            M_Tensors.append(ModeDoublets[final_path].conj())
                        elif path[1] == 'T':
                            M_Tensors.append(ModeTriplets[final_path].conj())
                        else:
                            raise ValueError('Failed to identiy type of the Mode')
                    elif path[0] == '1':
                        final_path = path[3:]+'_t'+str(self.SinkTime)
                        if path[1] == 'D':
                            M_Tensors.append(ModeDoublets[final_path])
                        elif path[1] == 'T':
                            M_Tensors.append(ModeTriplets[final_path])
                        else:
                            raise ValueError('Failed to identiy type of the Mode')
                    else:
                        raise ValueError('Failed to identify sink and source times')
            else:
                raise ValueError('Update the Method!')
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / 1024**2 
            print(f"Speicherverbrauch Nach M_Tensors und vor extractor: {mem:.1f} MB")
            extractor     = Perambulator_Laph(full_cluster[1][1][0])
            prmp_indx_In  = extractor['index_In']
            prmp_indx_Ou  = extractor['index_Out']
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / 1024**2 
            print(f"Speicherverbrauch nach extractor: {mem:.1f} MB")
            print(f' Index Infos: {DT_Index},{prmp_indx_In}->{prmp_indx_Ou}')
            print(f'Tensor Infos {[i.shape for i in [*M_Tensors]]} and {[i.shape for i in [*extractor["Tensor"]]]}')
            try:
                results   = torch.einsum(f'{DT_Index},{prmp_indx_In}->{prmp_indx_Ou}', *M_Tensors, *extractor['Tensor'])
            except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                raise TypeError('That should not happen!!!!!!')
            all_info = SpnFF_XTractor(full_cluster[1][1])
            results_factors = torch.stack([results[*i['Spin_Indices']] * i['Numerical_Factor'] for i in all_info])
            results_summed = torch.sum(results_factors, dim=0)
            clusters_with_kies_copy.append((full_cluster[0], results_summed))
            print(cntrctns_cntr)
            cntrctns_cntr+=1
        return clusters_with_kies_copy, self.WT_numerical_factors



            
#            for prmp_container in full_cluster[1][1][1:]:
#                peram_info = Perambulator_Setup(prmp_container)
#                if peram_info['index'] != prmp_indizes:
#                    raise ValueError('Something wrong with Perambulator_Extractor')
#                prmp_list.append(peram_info['Tensor'])
#            try:
#                Perambulators = torch.stack([Tensor_Product(Qs) for Qs in prmp_list], dim=0)
#                results      = torch.einsum(f'{DT_Index},Z{prmp_indizes}->Z', *M_Tensors, Perambulators)
#            except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
#                print(f"Skipping contractions for cluster {full_cluster[0][0]} and diagram(s) {full_cluster[0][1]} due to memory error:")
#                print(er)
#                print('__________')
#                continue
#            results = torch.sum(results, dim=0)
#            clusters_with_kies_copy.append((full_cluster[0], results))
#            print(cntrctns_cntr)
#            cntrctns_cntr+=1
        #return comparing_list
#        return clusters_with_kies_copy, self.WT_numerical_factors

    def TorchTractor_Old(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None):
        if (All_Perambulators is None):
            raise ValueError('The perambulators_dicts must be forwarded to TorchTractor as All_Perambulators = ...')
        if (ModeDoublets is None) and (ModeTriplets is None):
            er = 'TorchTractor must take as argument at least a ModeDoublet or a ModeTriplet.'
            raise ValueError(f'{er} one or both of the following arguments are missing: ModeDoublets = ..., ModeTriplets = ... ')
        def Perambulator_Setup(exp_prmp_container):
            Prmp_Indices = ''
            Prmp_Tensors = []
            seen_hadron = set()
            for perambulator in exp_prmp_container:
                Prmp_Indices += index_map[perambulator.getQ()]
                Prmp_Indices += index_map[perambulator.getQ_Bar()]
                s             = perambulator.getS() - 1
                s_Bar         = perambulator.getS_Bar() - 1
                p_left        = perambulator.getH()[0]
                p_right       = perambulator.getH_Bar()[0]
                prmp_flavor   = perambulator.getFlavor()
                num_factor    = 1
                if perambulator.getH() not in seen_hadron:
                    seen_hadron.add(perambulator.getH())
                    num_factor *= perambulator.getFF_H()
                if perambulator.getH_Bar() not in seen_hadron:
                    seen_hadron.add(perambulator.getH_Bar())
                    num_factor *= perambulator.getFF_H_Bar()
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
        print(f'{len(self.clusters_with_kies)} tensor contractions to be performed')
        cntrctns_cntr = 0
        for full_cluster in self.clusters_with_kies:
            Doublet_Triplet_Tensor_Info = full_cluster[1][0]['MDT_Info']
            DT_Index  = ','.join(full_cluster[1][0]['Mode_Index_Info'])
            if isinstance(Doublet_Triplet_Tensor_Info, tuple):
                print('One ModeDoublet/Triplet for a spin combination')
                M_Tensors = []
                for path in Doublet_Triplet_Tensor_Info:
                    if path[0] == '0':
                        final_path = path[3:]+'_t'+str(self.SourceTime)
                        if path[1] == 'D':
                            M_Tensors.append(ModeDoublets[final_path].conj())
                        elif path[1] == 'T':
                            M_Tensors.append(ModeTriplets[final_path].conj())
                        else:
                            raise ValueError('Failed to identiy type of the Mode')
                    elif path[0] == '1':
                        final_path = path[3:]+'_t'+str(self.SinkTime)
                        if path[1] == 'D':
                            M_Tensors.append(ModeDoublets[final_path])
                        elif path[1] == 'T':
                            M_Tensors.append(ModeTriplets[final_path])
                        else:
                            raise ValueError('Failed to identiy type of the Mode')
                    else:
                        raise ValueError('Failed to identify sink and source times')
            else:
                raise ValueError('Update the Method!')
            prmp_list     = []
            extractor     = Perambulator_Setup(full_cluster[1][1][0])
            prmp_indizes  = extractor['index']
            prmp_list.append(extractor['Tensor'])
            torchsum = True
            for prmp_container in full_cluster[1][1][1:]:
                peram_info = Perambulator_Setup(prmp_container)
                if peram_info['index'] != prmp_indizes:
                    raise ValueError('Something wrong with Perambulator_Extractor')
                prmp_list.append(peram_info['Tensor'])
            try:
                Perambulators = torch.stack([Tensor_Product(Qs) for Qs in prmp_list], dim=0)
                results      = torch.einsum(f'{DT_Index},Z{prmp_indizes}->Z', *M_Tensors, Perambulators)
            except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                print(f"Skipping contractions for cluster {full_cluster[0][0]} and diagram(s) {full_cluster[0][1]} due to memory error:")
                print(er)
                print('__________')
                continue
            results = torch.sum(results, dim=0)
            clusters_with_kies_copy.append((full_cluster[0], results))
            print(cntrctns_cntr)
            cntrctns_cntr+=1
        return clusters_with_kies_copy, self.WT_numerical_factors

    #commet_01
    '''
    def TorchTractor_old(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None):
        naive_clusters_with_kies = [((outer_key, inner_key), 
                                    Final_Perambulator_Container(prpm_container, self.hadron_product).getExplicit_Perambulator_Containers() ) 
                               for outer_key, inner_dict in self.clusters.items() 
                               for inner_key, prpm_container in inner_dict.items()]
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
                if hadron[0] == 0:
                    time          = 't'+str(self.SourceTime)
                    path          = mntm + '_' + disp + '_' + time
                    if self.hadron_type_mom_map[hadron]['T'] == 'M':
                        Mode_Tensors.append(ModeDoublets[path].conj())
                    elif self.hadron_type_mom_map[hadron]['T'] == 'B':
                        Mode_Tensors.append(ModeTriplets[path].conj())
                        Mode_Indices += index_map[hadron + (2,)]
                elif hadron[0] == 1:
                    time          = 't'+str(self.SinkTime)
                    path          = mntm + '_' + disp + '_' + time
                    if self.hadron_type_mom_map[hadron]['T'] == 'M':
                        Mode_Tensors.append(ModeDoublets[path])
                    elif self.hadron_type_mom_map[hadron]['T'] == 'B':
                        Mode_Tensors.append(ModeTriplets[path])
                        Mode_Indices += index_map[hadron + (2,)]
                else:
                    raise ValueError('Error 4')
                Mode_Indices = Mode_Indices + ','
            return {'index': Mode_Indices, 'Tensor': Mode_Tensors}
        #comment_02
        def Perambulator_Setup(exp_prmp_container):
            Prmp_Indices = ''
            Prmp_Tensors = []
            seen_hadron = set()
            for perambulator in exp_prmp_container:
                Prmp_Indices += index_map[perambulator.getQ()]
                Prmp_Indices += index_map[perambulator.getQ_Bar()]
                s             = perambulator.getS() - 1
                s_Bar         = perambulator.getS_Bar() - 1
                p_left        = perambulator.getH()[0]
                p_right       = perambulator.getH_Bar()[0]
                prmp_flavor   = perambulator.getFlavor()
                num_factor    = 1
                if perambulator.getH() not in seen_hadron:
                    seen_hadron.add(perambulator.getH())
                    num_factor *= perambulator.getFF_H()
                if perambulator.getH_Bar() not in seen_hadron:
                    seen_hadron.add(perambulator.getH_Bar())
                    num_factor *= perambulator.getFF_H_Bar()
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
        print(f'{len(naive_clusters_with_kies)} tensor contractions to be performed')
        cntrctns_cntr = 0
        comparing_list = {}
        for full_cluster in naive_clusters_with_kies:
            modes_info    = Modes_Setup(full_cluster[0][0], full_cluster[1][0])
            modes_indices = modes_info['index'][:-1]
            modes_tensors = modes_info['Tensor']
            prmp_list     = []
            extractor     = Perambulator_Setup(full_cluster[1][0])
            prmp_indizes  = extractor['index']
            prmp_list.append(extractor['Tensor'])
            torchsum = True
            for prmp_container in full_cluster[1][1:]:
                peram_info = Perambulator_Setup(prmp_container)
                if peram_info['index'] != prmp_indizes:
                    raise ValueError('Something wrong with Perambulator_Extractor')
                prmp_list.append(peram_info['Tensor'])
            try:
                Perambulators = torch.stack([Tensor_Product(Qs) for Qs in prmp_list], dim=0)
                comparing_list[full_cluster[0]] = {'DT_Index': modes_indices, 'prmp_indizes': prmp_indizes,
                                                   'M_Tensors': modes_tensors, 'Perambulators': Perambulators}
                results      = torch.einsum(f'{modes_indices},Z{prmp_indizes}->Z', *modes_tensors, Perambulators)
            except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                print(f"Skipping contractions for cluster {full_cluster[0][0]} and diagram(s) {full_cluster[0][1]} due to memory error:")
                print(er)
                print('__________')
                continue
            results = torch.sum(results, dim=0)
            clusters_with_kies_copy.append((full_cluster[0], results))
            print(cntrctns_cntr)
            cntrctns_cntr+=1
        #return comparing_list
        return clusters_with_kies_copy, self.WT_numerical_factors
    '''