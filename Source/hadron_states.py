import numpy as np
from contractions_handler import *

################################################################################################
###### Now    we  use the above classes to define some of the common  hadorn    operators ######
################################################################################################
fls = False
def Delta(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {3/2: hdrn(1, 'delta', mntm, 'u', 'u', 'u', barness = fls), 1/2:  hdrn(3**(1/2), 'delta', mntm, 'u', 'u', 'd', barness = fls),
           -1/2:  hdrn(3**(1/2), 'delta', mntm, 'u', 'd', 'd', barness = fls),-3/2: hdrn(1, 'delta', mntm, 'd', 'd', 'd', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of delta must be the isospin and it mus be in 3/2, 1/2, -1/2 or -3/2")
def Sigma(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {1: hdrn(1, 'Sigma', mntm, 'u', 'u', 's', barness = fls), 0:  hdrn(2**(1/2), 'Sigma', mntm, 'u', 'd', 's', barness = fls),
           -1:  hdrn(1, 'Sigma', mntm, 'd', 'd', 's', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of sigma must be the isospin and it mus be in 1, 0 or -1")
def Nucleon(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {1/2: hdrn(1, 'nucleon', mntm, 'u', 'u', 'd', barness = fls), -1/2:  hdrn(-1, 'nucleon', mntm, 'd', 'd', 'u', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of nucleon must be the isospin and it mus be in 1/2 or -1/2")
def Xi(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {1/2: hdrn(1, 'xi', mntm, 's', 's', 'u', barness = fls), -1/2:  hdrn(-1, 'xi', mntm, 's', 's', 'd', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of xi must be the isospin and it mus be in 1/2 or -1/2")
def Lambda(mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    return hdrn(1, 'lambda', mntm, 'u', 'd', 's', barness = fls)
def Omega(mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    return hdrn(1, 'omega', mntm, 's', 's', 's', barness = fls)
'''
def Kaon(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {'+': hdrn(1, 'K+', mntm, 'u', 'sB', barness = fls), 0:  hdrn(1, 'K0', mntm, 'd', 'sB', barness = fls)
            , '-':  hdrn(-1, 'K-', mntm, 'uB', 's', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of kaon must be the its type, i.e. +, - or 0 ")
'''
def Kaon(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {1/2: hdrn(1, 'K', mntm, 'sB', 'u', barness = fls), -1/2:  hdrn(1, 'K', mntm, 'sB', 'd', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of kaon must be the its type, i.e. +, - or 0 ")

def KaonC(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {1/2: hdrn(1, 'Kc', mntm, 'dB', 's', barness = fls), -1/2:  hdrn(-1, 'Kc', mntm, 'uB', 's', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of kaon must be the its type, i.e. +, - or 0 ")

def Pion(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {1: hdrn(1, 'Pi', mntm, 'dB', 'u', barness = fls), -1:  hdrn(-1, 'Pi', mntm, 'uB', 'd', barness = fls)
            , 0:  hdrn(1/2**(1/2), 'Pi', mntm, 'dB', 'd', barness = fls) + hdrn(-1/2**(1/2), 'Pi', mntm, 'uB', 'u', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of Pion must be the its type, i.e. 1, -1 or 0 ")


def sigma(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    return hdrn(1/(2**(1/2)), 'sigma', mntm, 'dB', 'd', barness = fls) + hdrn(1/(2**(1/2)), 'sigma', mntm, 'uB', 'u', barness = fls)

def Phi(mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    return hdrn(1, 'phi', mntm, 'sB', 's', barness = fls)
def DMeson(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {1: hdrn(1, 'D+', mntm, 'dB', 'c', barness = fls), -1:  hdrn(1, 'D-', mntm, 'cB', 'd', barness = fls)
            , 0:  hdrn(1, 'D0', mntm, 'uB', 'c', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of DMeson must be the its type, i.e. +, - or 0 ")

def Eta(mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    return hdrn(1/(2**(1/2)), 'Eta', mntm, 'uB', 'u', barness = fls) + hdrn(1/(2**(1/2)), 'Eta', mntm, 'dB', 'd', barness = fls)




def normalize_function(coefficients):
    return 1 / np.sqrt(sum(c**2 for c in coefficients))

def TwoHadronAnnihilation(rep=None, I=None, I3=None):
    if list(rep) == [1/2, 1/2]:
        # Equation 7.13: 1/2 ⊗ 1/2
        if I == 0 and I3 == 0:
            state = [(1/2, -1/2), (-1/2, 1/2)]
            super_position_factors = [1, -1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1 and I3 == 1:
            state = [(1/2, 1/2)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1 and I3 == 0:
            state = [(1/2, -1/2), (-1/2, 1/2)]
            super_position_factors = [1, 1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1 and I3 == -1:
            state = [(-1/2, -1/2)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
            
    elif list(rep) == [1/2, 1]:
        # Equation 7.14: 1/2 ⊗ 1
        if I == 1/2 and I3 == 1/2:
            state = [(1/2, 0), (-1/2, 1)]
            super_position_factors = [1, -np.sqrt(2)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1/2 and I3 == -1/2:
            state = [(-1/2, 0), (1/2, -1)]
            super_position_factors = [1, -np.sqrt(2)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 3/2 and I3 == 3/2:
            state = [(1/2, 1)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 3/2 and I3 == 1/2:
            state = [(-1/2, 1), (1/2, 0)]
            super_position_factors = [1, np.sqrt(2)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 3/2 and I3 == -1/2:
            state = [(1/2, -1), (-1/2, 0)]
            super_position_factors = [1, np.sqrt(2)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 3/2 and I3 == -3/2:
            state = [(-1/2, -1)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
            
    elif list(rep) == [1, 1]:
        # Equation 7.15: 1 ⊗ 1
        if I == 0 and I3 == 0:
            state = [(1, -1), (0, 0), (-1, 1)]
            super_position_factors = [1, -1, 1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1 and I3 == 1:
            state = [(1, 0), (0, 1)]
            super_position_factors = [1, -1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1 and I3 == 0:
            state = [(1, -1), (-1, 1)]
            super_position_factors = [1, -1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1 and I3 == -1:
            state = [(0, -1), (-1, 0)]
            super_position_factors = [1, -1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == 2:
            state = [(1, 1)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == 1:
            state = [(1, 0), (0, 1)]
            super_position_factors = [1, 1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == 0:
            state = [(1, -1), (0, 0), (-1, 1)]
            super_position_factors = [1, 2, 1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == -1:
            state = [(0, -1), (-1, 0)]
            super_position_factors = [1, 1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == -2:
            state = [(-1, -1)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
            
    elif list(rep) == [1/2, 3/2]:
        # Equation 7.16: 1/2 ⊗ 3/2
        if I == 1 and I3 == 1:
            state = [(1/2, 1/2), (-1/2, 3/2)]
            super_position_factors = [1, -np.sqrt(3)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1 and I3 == 0:
            state = [(1/2, -1/2), (-1/2, 1/2)]
            super_position_factors = [1, -1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1 and I3 == -1:
            state = [(1/2, -3/2), (-1/2, -1/2)]
            super_position_factors = [np.sqrt(3), -1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == 2:
            state = [(1/2, 3/2)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == 1:
            state = [(1/2, -3/2), (-1/2, -1/2)]
            super_position_factors = [1, np.sqrt(3)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == 0:
            state = [(1/2, -1/2), (-1/2, 1/2)]
            super_position_factors = [1, 1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == -1:
            state = [(1/2, 1/2), (-1/2, 3/2)]
            super_position_factors = [np.sqrt(3), 1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 2 and I3 == -2:
            state = [(-1/2, -3/2)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
            
    elif list(rep) == [1, 3/2]:
        # Equation 7.17: 1 ⊗ 3/2
        if I == 1/2 and I3 == 1/2:
            state = [(1, -1/2), (0, 1/2), (-1, 3/2)]
            super_position_factors = [1, -np.sqrt(2), np.sqrt(3)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 1/2 and I3 == -1/2:
            state = [(1, -3/2), (0, -1/2), (-1, 1/2)]
            super_position_factors = [np.sqrt(3), -np.sqrt(2), 1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 3/2 and I3 == 3/2:
            state = [(1, 1/2), (0, 3/2)]
            super_position_factors = [np.sqrt(2), -np.sqrt(3)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 3/2 and I3 == 1/2:
            state = [(1, -1/2), (0, 1/2), (-1, 3/2)]
            super_position_factors = [4, -np.sqrt(2), -2*np.sqrt(3)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 3/2 and I3 == -1/2:
            state = [(1, -3/2), (0, -1/2), (-1, 1/2)]
            super_position_factors = [2*np.sqrt(3), np.sqrt(2), -4]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 3/2 and I3 == -3/2:
            state = [(0, -3/2), (-1, -1/2)]
            super_position_factors = [np.sqrt(3), -np.sqrt(2)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 5/2 and I3 == 5/2:
            state = [(1, 3/2)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 5/2 and I3 == 3/2:
            state = [(1, 1/2), (0, 3/2)]
            super_position_factors = [np.sqrt(3), np.sqrt(2)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 5/2 and I3 == 1/2:
            state = [(1, -1/2), (0, 1/2), (-1, 3/2)]
            super_position_factors = [3, 3*np.sqrt(2), np.sqrt(3)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 5/2 and I3 == -1/2:
            state = [(1, -3/2), (0, -1/2), (-1, 1/2)]
            super_position_factors = [np.sqrt(3), 3*np.sqrt(2), 3]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 5/2 and I3 == -3/2:
            state = [(0, -3/2), (-1, -1/2)]
            super_position_factors = [np.sqrt(2), np.sqrt(3)]
            overall_normalization = normalize_function(super_position_factors)
        elif I == 5/2 and I3 == -5/2:
            state = [(-1, -3/2)]
            super_position_factors = [1]
            overall_normalization = normalize_function(super_position_factors)
    else:
        raise ValueError(f"Representation {rep} not implemented")
    
    return state, super_position_factors, overall_normalization


colins_reps = [[1/2, 1/2], [1/2, 1], [1, 1], [1/2, 3/2], [1, 3/2]]
State_Detector = {'Delta': 3/2, 'Sigma': 1, 'Nucleon': 1/2, 'Xi': 1/2, 'Lambda': 0, 'Omega': 0,
                  'Kaon': 1/2, 'KaonC': 1/2, 'Pion': 1, 'sigma': 0, 'Phi': 0, 'DMeson': 1, 'Eta': 0}

def twoHO(rep=None, I=None, I3=None, A=None, B=None):
    name_A = A.__name__
    name_B = B.__name__
    IA     = rep[0]
    IB     = rep[1]
    if State_Detector[name_A] == IA and  State_Detector[name_B] == IB:
        if list(rep) in colins_reps:
            state, sprpstnFF, Overall_N = TwoHadronAnnihilation(rep=rep, I=I, I3=I3)
            N = len(state)
            if N != len(sprpstnFF):
                raise ValueError('Failed to identify the two Hadrons')
            hadron = [Overall_N * sprpstnFF[i] * A(state[i][0]) * B(state[i][1]) for i in range(N)]
            hadrons = hadron[0]
            for i in range(1, N):
                hadrons += hadron[i]
            return hadrons
        elif list(reversed(list(rep))) in colins_reps:
            state, sprpstnFF, Overall_N =  TwoHadronAnnihilation(rep=list(reversed(list(rep))), I=I, I3=I3)
            N = len(state)
            if N != len(sprpstnFF):
                raise ValueError('Failed to identify the two Hadrons')
            hadron = [Overall_N * sprpstnFF[i] * A(state[i][1]) * B(state[i][0]) for i in range(N)]
            hadrons = hadron[0]
            for i in range(1, N):
                hadrons += hadron[i]
            return hadrons
        else:
            raise ValueError('Failed to identify the two Hadrons')
    else:
        raise ValueError(f'For rep {rep}, First Hadron A is expected to be {IA} while Second Hadron B {IB}')