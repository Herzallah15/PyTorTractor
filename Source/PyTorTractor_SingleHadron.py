from PyTorDefinitions import *
from PyTor_S_Definitions import *
from SubDiagraming_jit import *
import opt_einsum as oe
import h5py
import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple, Any
import os, psutil

class PyCorrTorch_SingleCor:
    def __init__(self, SinkTime = None, SourceTime = None, current_time = None, 
                 Hadrons = None, Path_Wicktract = None, Wick_subpath = None):
        self.SinkTime          = SinkTime
        self.SourceTime        = SourceTime
        self.current_time       = current_time
        self.Hadrons           = Hadrons
        self.Path_Wicktract    = Path_Wicktract
        self.Wick_subpath      = Wick_subpath
        hadron_momentum_map = {}
        for hadron in self.Hadrons:
            hadron_momentum_map[hadron.getHadron_Position()] = tuple(hadron.getMomentum())
        self.hadron_momentum_map = hadron_momentum_map

        # part which is commented out. Called PART_OUTCOMMENT_0
        # Cluster the Diagrams
        self.clusters, self.WT_numerical_factors = cluster_extractor(Path_Diagrams = self.Path_Wicktract, Sub_Path = self.Wick_subpath)


        fullmap_hadrons = {}
        for one_hadron in self.Hadrons:
            fullmap_hadrons[one_hadron.getHadron_Position()] = one_hadron
        self.fullmap_hadrons = fullmap_hadrons

        # SpinStructure Combinations between the hadrons
        hadron_combi_map = {}
        for clstr in self.clusters:
            hadron_combi_map[clstr] = hadron_info_multiplier(*co_to_Hadorn_co(clstr, self.fullmap_hadrons))
        self.hadron_combi_map = hadron_combi_map
        print('All combinations of hadron structures coefficients were generated')

        # Sort the hadrons to baryons and mesons so later it becomes easier to extract MT and MD
        hadron_type_mom_map = {}
        for hdrn in self.Hadrons:
            hadron_type_mom_map[hdrn.getHadron_Position()] = {'T': hdrn_type(hdrn.getHadron_Type()), 'P': hdrn.getMomentum_Value()}
        self.hadron_type_mom_map = hadron_type_mom_map


        self.clusters_with_kies = [((outer_key, inner_key),
                                    [Final_Perambulator_Container(prpm_container, pick_combis(prpm_container.getHadrons(),self.hadron_combi_map)
                                                                 ).getModeInfos(),
                                    Final_Perambulator_Container(prpm_container, pick_combis(prpm_container.getHadrons(),self.hadron_combi_map)
                                                                ).getExplicit_Perambulator_Containers()] )
                               for outer_key, inner_dict in self.clusters.items()
                               for inner_key, prpm_container in inner_dict.items()]
        print('Each cluster is now splitted into many clusters with various explicit spin combinations')

    def TorchTractor_SingleCor(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None, all_SG_perambulators = None,
                              optimal_path = True):
        if (All_Perambulators is None):
            raise ValueError('The perambulators_dicts must be forwarded to TorchTractor as All_Perambulators = ...')
        if (ModeDoublets is None) and (ModeTriplets is None):
            er = 'TorchTractor must take as argument at least a ModeDoublet or a ModeTriplet.'
            raise ValueError(f'{er} One or both of the following arguments are missing: ModeDoublets = ..., ModeTriplets = ... ')
        print(f'{len(self.clusters_with_kies)} tensor contractions to be performed')
        cntrctns_cntr = 0
        clusters_with_kies_copy = []
        for full_cluster in self.clusters_with_kies:
            all_cluter_info   = full_cluster[1][1]
            all_Modes_Paths   = full_cluster[1][0]['MDT_Info']
            mode_indices      = ','.join(full_cluster[1][0]['Mode_Index_Info'])
            Ps_indices, Mode_Indices, Stacked_Ps, Stacked_Ms  = Perambulator_Mode_Handler(Full_Cluster = all_cluter_info,
                                                                                          All_Mode_Info = all_Modes_Paths,
                                                                                          snktime = self.SinkTime, srctime=self.SourceTime,
                                                                                          Mode_Unsplitted_Index = mode_indices,
                                                                                          Prmbltr = All_Perambulators, ModeD = ModeDoublets,
                                                                                          ModeT = ModeTriplets,
                                                                                          all_SG_perambulators = all_SG_perambulators,
                                                                                          Hadron_Momenta = self.hadron_momentum_map,
                                                                                          current_time= self.current_time)
            normalized_pattern = NormalizePattern(Mode_Indices, Ps_indices, Ps_indices[:2])
            shapes = tuple([tuple(t.shape) for t in [*Stacked_Ms, *Stacked_Ps]])
            cache_key = f"{normalized_pattern}_{shapes}"
            if optimal_path:
                if path_exists_in_hdf5(cache_key):
                    path = load_path_from_hdf5(cache_key)
                else:
                    path_info = oe.contract_path(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Stacked_Ms, *Stacked_Ps, optimize='optimal')
                    path = path_info[0]
                    save_path_to_hdf5(cache_key, path)
                try:
                    res = oe.contract(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Stacked_Ms, *Stacked_Ps, optimize=path).sum()
                except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                    raise TypeError('That should not happen!!!!!!')
            else:
                os.makedirs("jit_cache", exist_ok=True)
                hash_key = hashlib.md5(cache_key.encode()).hexdigest()
                cache_file = f"jit_cache/jit_{hash_key}.pt"
                if os.path.exists(cache_file):
                    traced_func = torch.jit.load(cache_file)
                else:
                    einsum_str = f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}'
                    def einsum_contract(*tensors):
                        return torch.einsum(einsum_str, *tensors)
                    traced_func = torch.jit.trace(einsum_contract, (*Stacked_Ms, *Stacked_Ps))
                    traced_func.save(cache_file)
                try:
                    res = traced_func(*Stacked_Ms, *Stacked_Ps).sum()
                except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                    raise TypeError('That should not happen!!!!!!')
            clusters_with_kies_copy.append((full_cluster[0], res))
            #print(cntrctns_cntr)
            cntrctns_cntr+=1
        return clusters_with_kies_copy, self.WT_numerical_factors
    def TorchTractor_SingleCor_OnlyPStack(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None):
        if (All_Perambulators is None):
            raise ValueError('The perambulators_dicts must be forwarded to TorchTractor as All_Perambulators = ...')
        if (ModeDoublets is None) and (ModeTriplets is None):
            er = 'TorchTractor must take as argument at least a ModeDoublet or a ModeTriplet.'
            raise ValueError(f'{er} One or both of the following arguments are missing: ModeDoublets = ..., ModeTriplets = ... ')
        print(f'{len(self.clusters_with_kies)} tensor contractions to be performed')
        cntrctns_cntr = 0
        clusters_with_kies_copy = []
        for full_cluster in self.clusters_with_kies:
            all_cluter_info   = full_cluster[1][1]
            all_Modes_Paths   = full_cluster[1][0]['MDT_Info']
            mode_indices      = ','.join(full_cluster[1][0]['Mode_Index_Info'])
            Mode_P_Info       = Perambulator_Mode_Handler_PStacked(Full_Cluster = all_cluter_info, All_Mode_Info = all_Modes_Paths,
                                      snktime = self.SinkTime, srctime=self.SourceTime,
                                      Prmbltr = All_Perambulators, ModeD = ModeDoublets, ModeT = ModeTriplets)
            Unique_Mode = Mode_P_Info['Unique_Paths']
            stckd_P_idx = Mode_P_Info['Ps_Stacked_Indices']
            Mode_P_Info = Mode_P_Info['Mode_P']
            N = len(stckd_P_idx)
            i = 0
            try:
                res = torch.einsum(f'{mode_indices},{stckd_P_idx[i]}->{stckd_P_idx[i][0]}',
                                   *Mode_P_Info[Unique_Mode[i]]['ExplicitModes'],*Mode_P_Info[Unique_Mode[i]]['ExplicitPerambulators']).sum()
                for i in range(1, N):
                    res += torch.einsum(f'{mode_indices},{stckd_P_idx[i]}->{stckd_P_idx[i][0]}',
                                       *Mode_P_Info[Unique_Mode[i]]['ExplicitModes'],*Mode_P_Info[Unique_Mode[i]]['ExplicitPerambulators']).sum()
            except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                raise TypeError('That should not happen!!!!!!')
            clusters_with_kies_copy.append((full_cluster[0], res))
            print(cntrctns_cntr)
            cntrctns_cntr+=1
        return clusters_with_kies_copy, self.WT_numerical_factors

    def TorchTractor_Single_Naive(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None):
        if (All_Perambulators is None):
            raise ValueError('The perambulators_dicts must be forwarded to TorchTractor as All_Perambulators = ...')
        if (ModeDoublets is None) and (ModeTriplets is None):
            er = 'TorchTractor must take as argument at least a ModeDoublet or a ModeTriplet.'
            raise ValueError(f'{er} One or both of the following arguments are missing: ModeDoublets = ..., ModeTriplets = ... ')
        print(f'{len(self.clusters_with_kies)} tensor contractions to be performed')
        cntrctns_cntr = 0
        clusters_with_kies_copy = []
        for full_cluster in self.clusters_with_kies:
            prmp_indx     = Perambulator_Laph(All_Perambulators, full_cluster[1][1][0], self.SinkTime, self.SourceTime)['index']
            res_dtype     = Perambulator_Laph(All_Perambulators, full_cluster[1][1][0], self.SinkTime, self.SourceTime)['Tensor'][0].dtype
            res_device    = Perambulator_Laph(All_Perambulators, full_cluster[1][1][0], self.SinkTime, self.SourceTime)['Tensor'][0].device
            DT_Index      = ','.join(full_cluster[1][0]['Mode_Index_Info'])
            DT_AllPaths   = full_cluster[1][0]['MDT_Info']
            result = torch.tensor(0.0, dtype = res_dtype, device = res_device)
            N = len(full_cluster[1][1])
            if N != len(DT_AllPaths):
                raise ValueError('Failed to combin the Modes with the Perambulators!')
            for i in range(N):
                M_Tensors = MDT_Laph(MDT_Info = DT_AllPaths[i], snktime = self.SinkTime, 
                                     srctime = self.SourceTime, ModeD = ModeDoublets, ModeT = ModeTriplets)
                extractor = Perambulator_Laph(All_Perambulators, full_cluster[1][1][i], self.SinkTime, self.SourceTime)
                if extractor['index'] != prmp_indx:
                    raise ValueError('Inconsistency with Perambulator indices')
                try:
                    result += torch.einsum(f'{DT_Index},{prmp_indx}->', *M_Tensors, *extractor['Tensor'])
                except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                    raise TypeError('That should not happen!!!!!!')
            clusters_with_kies_copy.append((full_cluster[0], result))
            print(cntrctns_cntr)
            cntrctns_cntr+=1
        return clusters_with_kies_copy, self.WT_numerical_factors
'''
    def TorchTractor_Old(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None):
        if (All_Perambulators is None):
            raise ValueError('The perambulators_dicts must be forwarded to TorchTractor as All_Perambulators = ...')
        if (ModeDoublets is None) and (ModeTriplets is None):
            er = 'TorchTractor must take as argument at least a ModeDoublet or a ModeTriplet.'
            raise ValueError(f'{er} One or both of the following arguments are missing: ModeDoublets = ..., ModeTriplets = ... ')
        print(f'{len(self.clusters_with_kies)} tensor contractions to be performed')
        cntrctns_cntr = 0
        def Perambulator_Laph(exp_prmp_container):
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
            return {'index_In': Prmp_Indices_In[:-1], 'index_Out': Prmp_Indices_Out, 'Tensor': Prmp_Tensors}
        clusters_with_kies_copy = []
        for full_cluster in self.clusters_with_kies:
            Doublet_Triplet_Tensor_Info = full_cluster[1][0]['MDT_Info']
            DT_Index  = ','.join(full_cluster[1][0]['Mode_Index_Info'])
            if isinstance(Doublet_Triplet_Tensor_Info, tuple):
                print('One ModeDoublet/Triplet for all spin combination')
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
            extractor     = Perambulator_Laph(full_cluster[1][1][0])
            prmp_indx_In  = extractor['index_In']
            prmp_indx_Ou  = extractor['index_Out']
            #print(f' Index Infos: {DT_Index},{prmp_indx_In}->{prmp_indx_Ou}')
            #print(f'Tensor Infos {[i.shape for i in [*M_Tensors]]} and {[i.shape for i in [*extractor["Tensor"]]]}')
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
'''
