import numpy as np
import torch
from torch import nn
import h5py
import copy
#In this context I plan to have Q of the form np.array([1,0,0]) and H of the form np.array([1,0])
class Perambulator:
    def __init__(self, Q, Q_Bar, H, H_Bar, Flavor):
        self.Q      = Q
        self.Q_Bar  = Q_Bar
        self.H      = H
        self.H_Bar  = H_Bar
        self.Flavor = Flavor
    def getQ(self):
        return self.Q
    def getQ_Bar(self):
        return self.Q_Bar
    def getH(self):
        return self.H
    def getH_Bar(self):
        return self.H_Bar
    def getFlvr(self):
        return self.Flavor
    def show(self):
        return np.array([self.getQ(), self.getQ_Bar()])
    def getPartners(self):
        partners = set()
        partners.add(tuple(self.getH().tolist()))
        partners.add(tuple(self.getH_Bar().tolist()))
        return partners
    def __eq__(self, other):
        if isinstance(other, Perambulator):
            return np.all(self.getQ() == other.getQ()) and np.all(self.getQ_Bar() == other.getQ_Bar())
        else:
            raise TypeError('Perambulator-typed object can be compared with only perambulator-typed one!')
    def __mul__(self, other):
        if isinstance(other, Perambulator):
            return Perambulator_Container(*[self, other])
        elif isinstance(other, Perambulator_Container):
            return Perambulator_Container(*([self] + other.getPerambulators()))
    def __rmul__(self, other):
        if isinstance(other, Perambulator_Container):
            return self * other
        else:
            raise TypeError('Undefined * operator with Perambulator-typed object')
class Perambulator_Container:
    def __init__(self, *perambulators):
        self.perambulators = perambulators
    def getPerambulators(self):
        return list(self.perambulators)
    def show(self):
        dgrm = []
        for i in self.getPerambulators():
            dgrm.append(i.show())
        return np.array(dgrm)
    def __mul__(self, other):
        if isinstance(other, Perambulator):
            return Perambulator_Container(*([other] + self.getPerambulators()))
        elif isinstance(other, Perambulator_Container):
            return Perambulator_Container(*(self.getPerambulators() + other.getPerambulators()))
    def __rmul__(self, other):
        if isinstance(other, Perambulator):
            return self * other
        else:
            raise TypeError('Undefined * operator with Perambulator_Container object!')
    def __eq__(self, other):
        if isinstance(other, Perambulator_Container):
            set1  = self.getPerambulators()
            set2  = other.getPerambulators()
            seen = set()
            if len(set1) != len(set2):
                return False
            else:
                for prmp1 in set1:
                    found = False
                    for i, prmp2 in enumerate(set2):
                        if i in seen:
                            continue
                        if prmp1 == prmp2:
                            seen.add(i)
                            found = True
                            break
                    if not found:
                        break
                return len(seen) == len(set1)
        else:
            raise TypeError('A Perambulator_Container-typed object can be only compared with a Perambulator_Container-typed objects')
class Diagram:
    def __init__(self, dgrmn, *perambulators):
        self.dgrmn         = dgrmn
        self.perambulators = perambulators
    def getPerambulators(self):
        return list(self.perambulators)
    def organize(self):
        perambulators_product = self.getPerambulators()
        clusters    = [[i.getPartners(), [i]] for i in perambulators_product]
        while True:
            new_cluster = []
            used = set()
            for i1, clust1 in enumerate(clusters):
                clus1 = clust1[0]
                topo1 = clust1[1]
                if i1 in used:
                    continue
                merged_cluster = copy.deepcopy(clus1)
                merged_topo    = copy.deepcopy(topo1)
                used.add(i1)
                for i2, clust2 in enumerate(clusters[i1+1:], i1+1):
                    clus2 = clust2[0]
                    topo2 = clust2[1]
                    if (merged_cluster & clus2) and (i2 not in used):
                        merged_cluster |= clus2
                        merged_topo.extend(topo2)
                        used.add(i2)
                new_cluster.append([merged_cluster, merged_topo])
            if len(new_cluster) == len(clusters):
                break
            clusters = new_cluster
        cluster_map = {}
        for clust in clusters:
            clus                     = clust[0]
            topologies               = clust[1]
            tp1 = topologies[0]
            for topology in topologies[1:]:
                tp1 *= topology
            topologies = tp1 if isinstance(tp1, Perambulator_Container) else Perambulator_Container(tp1)
            cluster_map[tuple(clus)] = {(self.dgrmn,): topologies}
        return cluster_map
#organized_diagrams is a list of objects of the form diagram.organize()
class Diagram_Container:
    def __init__(self, organized_diagrams):
        self.organized_diagrams = organized_diagrams
    def getOrganized_Diagrams(self):
        return self.organized_diagrams
    def do_clustering(self):
#first do it for two submaps
# The following function takes two maps of the form {(diagram_numbers): Toplogies, ... }
# and combines them with each others! In 
        def inner_list_combiner(submap1, submap2):
            seen       = set()
            fnl_submap = {}
            for dgrm_nmr_lst1 in submap1:
                fnl_hdrnlst = copy.deepcopy(dgrm_nmr_lst1)
                for i, dgrm_nmr_lst2 in enumerate(submap2):
                    if i in seen:
                        continue
                    if submap1[dgrm_nmr_lst1] == submap2[dgrm_nmr_lst2]:
                        seen.add(i)
                        fnl_hdrnlst += dgrm_nmr_lst2
                fnl_submap[fnl_hdrnlst] = submap1[dgrm_nmr_lst1]
            for i, dgrm_nmr_lst2 in enumerate(submap2):
                if i not in seen:
                    fnl_submap[dgrm_nmr_lst2] = submap2[dgrm_nmr_lst2]
            return fnl_submap
# and do comparision with a single submap with iteself to just in case make sure, that no equal topologies appear in the same submap
        def single_map(submap):
            seen       = set()
            fnl_submap = {}
            for i1, dgrm_nmr_lst1 in enumerate(submap):
                if i1 in seen:
                    continue
                fnl_hdrnlst = copy.deepcopy(dgrm_nmr_lst1)
                for i2, dgrm_nmr_lst2 in enumerate(submap):
                    if i2 > i1:
                        if i2 in seen:
                            continue
                        if submap[dgrm_nmr_lst1] == submap[dgrm_nmr_lst2]:
                            seen.add(i2)
                            fnl_hdrnlst += dgrm_nmr_lst2
                fnl_submap[fnl_hdrnlst] = submap[dgrm_nmr_lst1]
            return fnl_submap  
#third do it for two maps
        def maps_combiner(map1, map2):
            seen    = set()
            fnl_map = {}
            for cluster1 in map1:
                fnl_cluster = copy.deepcopy(map1[cluster1])
                for i, cluster2 in enumerate(map2):
                    if i in seen:
                        continue
                    if set(cluster1) == set(cluster2):
                        seen.add(i)
                        fnl_cluster = copy.deepcopy(inner_list_combiner(map1[cluster1], map2[cluster2]))
                        break
                fnl_map[cluster1] = fnl_cluster
            for i, cluster2 in enumerate(map2):
                if i not in seen:
                    fnl_map[cluster2] = map2[cluster2]
            return fnl_map
#Finally, combine all together
        disjoint_clusters = self.getOrganized_Diagrams()[0]
        all_clusters      = self.getOrganized_Diagrams()[1:]
        for map1 in all_clusters:
            disjoint_clusters = maps_combiner(disjoint_clusters, map1)
        final_cluster_map = {}
        for cluster in disjoint_clusters:
            final_cluster_map[cluster] = single_map(disjoint_clusters[cluster])
        return final_cluster_map