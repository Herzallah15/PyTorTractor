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
        self._path_cache       = {}
        self._failed_patterns  = set()
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
        #print('All combinations of hadron structures coefficients were generated')

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
        #print('Each cluster is now splitted into many clusters with various explicit spin combinations')
    def TorchTractor_SingleCor(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None, all_SG_perambulators = None):
        if (All_Perambulators is None):
            raise ValueError('The perambulators_dicts must be forwarded to TorchTractor as All_Perambulators = ...')
        if (ModeDoublets is None) and (ModeTriplets is None):
            er = 'TorchTractor must take as argument at least a ModeDoublet or a ModeTriplet.'
            raise ValueError(f'{er} One or both of the following arguments are missing: ModeDoublets = ..., ModeTriplets = ... ')
        #print(f'{len(self.clusters_with_kies)} tensor contractions to be performed')
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
            n_tensors = len(Stacked_Ms) + len(Stacked_Ps)
            device = Stacked_Ps[0].device
            if n_tensors <= 8:
                res = oe.contract(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Stacked_Ms, *Stacked_Ps).sum()
                clusters_with_kies_copy.append((full_cluster[0], res))
                cntrctns_cntr+=1
                del Stacked_Ps
                del Stacked_Ms
                continue
            normalized_pattern = NormalizePattern(Mode_Indices, Ps_indices, Ps_indices[:2])
            shapes = tuple([tuple(t.shape) for t in [*Stacked_Ms, *Stacked_Ps]])
            cache_key = f"{normalized_pattern}_{shapes}"
            if cache_key not in self._path_cache and cache_key not in self._failed_patterns:
                path_info = oe.contract_path(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Stacked_Ms, *Stacked_Ps, optimize='random-greedy-128')
                self._path_cache[cache_key] = path_info[0]
            try:
                if cache_key in self._failed_patterns:
                    raise MemoryError("Known failed pattern")
                else:
                    path = self._path_cache[cache_key]
                res = oe.contract(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Stacked_Ms, *Stacked_Ps, optimize=path).sum()
                clusters_with_kies_copy.append((full_cluster[0], res))
                cntrctns_cntr+=1
                del Stacked_Ms
                del Stacked_Ps
            except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                chunk_1_achieved = False
                self._failed_patterns.add(cache_key)
                if True:
                    Ny = Stacked_Ms[0].shape[0]
                    Nz = Stacked_Ps[0].shape[1]
                    chunk_size_y = Ny
                    chunk_size_z = Nz
                    success = False
                    while not success:
                        if chunk_size_y == 1 and chunk_size_z == 1:
                            chunk_1_achieved = True
                        try:
                            res = torch.tensor(0, device=device, dtype=Stacked_Ms[0].dtype)
                            for i in range(0, Ny, chunk_size_y):
                                i_end = min(i + chunk_size_y, Ny)
                                for j in range(0, Nz, chunk_size_z):
                                    j_end = min(j + chunk_size_z, Nz)
                                    Ms_chunk = []
                                    for mode_tensor in Stacked_Ms:
                                        if mode_tensor.ndim == 4:
                                            chunked_mode = mode_tensor[i:i_end, :, :, :]
                                        elif mode_tensor.ndim == 3:
                                            chunked_mode = mode_tensor[i:i_end, :, :]
                                        else:
                                            raise ValueError(f'Unrecognized Mode shape: {mode_tensor.shape}')
                                        Ms_chunk.append(chunked_mode)
                                    Ps_chunk = []
                                    for peram_tensor in Stacked_Ps:
                                        if peram_tensor.ndim == 4:
                                            chunked_peram = peram_tensor[i:i_end, j:j_end, :, :]
                                        else:
                                            raise ValueError(f'Unrecognized Perambulator shape: {peram_tensor.shape}')
                                        Ps_chunk.append(chunked_peram)
                                    res_chunk = oe.contract(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Ms_chunk, *Ps_chunk)
                                    res += res_chunk.sum()
                                    del Ms_chunk, Ps_chunk
                            clusters_with_kies_copy.append((full_cluster[0], res))
                            cntrctns_cntr+=1
                            success = True
                            del Stacked_Ms
                            del Stacked_Ps
                        except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                            torch.cuda.empty_cache()
                            if chunk_size_y >= chunk_size_z:
                                chunk_size_y = max(chunk_size_y // 2, 1)
                            else:
                                chunk_size_z = max(chunk_size_z // 2, 1)
                            if chunk_1_achieved:
                                raise RuntimeError('Contraction failed, although minimal operations were performed')
        return clusters_with_kies_copy, self.WT_numerical_factors
    def TorchTractor_SingleCorWorking(self, All_Perambulators = None, ModeDoublets = None, ModeTriplets = None, all_SG_perambulators = None,
                              optimal_path = None):
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
            if optimal_path is True:
                if path_exists_in_hdf5(cache_key):
                    path = load_path_from_hdf5(cache_key)
                else:
                    path_info = oe.contract_path(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Stacked_Ms, *Stacked_Ps, optimize='auto')
                    path = path_info[0]
                    save_path_to_hdf5(cache_key, path)
                try:
                    res = oe.contract(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Stacked_Ms, *Stacked_Ps, optimize=path).sum()
                except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                    try:
                        Ny = Stacked_Ms[0].shape[0]
                        Nz = Stacked_Ps[0].shape[1]
                        chunk_size_y = 1#10 if Ny > 10 else 1
                        chunk_size_z = 1#10 if Nz > 10 else 1
                        res = 0.0
                        for i in range(0, Ny, chunk_size_y):
                            i_end = min(i + chunk_size_y, Ny)
                            for j in range(0, Nz, chunk_size_z):
                                j_end = min(j + chunk_size_z, Nz)
                                Ms_chunk = []
                                for mode_tensor in Stacked_Ms:
                                    if mode_tensor.ndim == 4:
                                        chunked_mode = mode_tensor[i:i_end, :, :, :]
                                    elif mode_tensor.ndim == 3:
                                        chunked_mode = mode_tensor[i:i_end, :, :]
                                    else:
                                        raise ValueError(f'Unrecognized Mode shape: {mode_tensor.shape}')
                                    Ms_chunk.append(chunked_mode)
                                Ps_chunk = []
                                for peram_tensor in Stacked_Ps:
                                    if peram_tensor.ndim == 4:
                                        chunked_peram = peram_tensor[i:i_end, j:j_end, :, :]
                                    else:
                                        raise ValueError(f'Unrecognized Perambulator shape: {peram_tensor.shape}')
                                    Ps_chunk.append(chunked_peram)
                                res_chunk = oe.contract(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Ms_chunk, *Ps_chunk, optimize='auto')
                                res += res_chunk.sum()
                    except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                        raise TypeError('That should not happen!!!!!!')
            elif optimal_path is False:
                print('Warining: You are not using the optimal path!')
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
            else:
                try:
                    res = torch.einsum(f'{Mode_Indices},{Ps_indices}->{Ps_indices[:2]}', *Stacked_Ms, *Stacked_Ps).sum()
                except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as er:
                    raise TypeError('That should not happen!!!!!!')
            clusters_with_kies_copy.append((full_cluster[0], res))
            #print(cntrctns_cntr)
            cntrctns_cntr+=1
        return clusters_with_kies_copy, self.WT_numerical_factors