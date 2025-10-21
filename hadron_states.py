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




def TwoHadronAnnihilation(rep=None, I=None, I3=None, A=None, B=None):
    """
    Compute two-hadron annihilation operators for given isospin representations.
    
    Parameters:
    -----------
    rep : list
        Representation as [I_A, I_B] where I_A and I_B are isospin values
    I : int
        Total isospin quantum number
    I3 : int
        Isospin projection quantum number
    A : function
        Annihilation operator A (left operator)
    B : function
        Annihilation operator B (right operator)
        
    Returns:
    --------
    Combined operator expression
    
    Note: A is always on the left, B is always on the right (annihilation operators)
    """
    if list(rep) == [1/2, 1/2]:        
        if I == 0 and I3 == 0:
            return (B(-1/2) * A(1/2) - B(1/2) * A(-1/2))/np.sqrt(2)
        elif I == 1 and I3 == 1:
            return B(1/2) * A(1/2)
        elif I == 1 and I3 == 0:
            return (B(-1/2) * A(1/2) + B(1/2) * A(-1/2))/np.sqrt(2)
        elif I == 1 and I3 == -1:
            return B(-1/2) * A(-1/2)
            
    elif list(rep) == [1/2, 1]:
        # Equation 7.14: 1/2 ⊗ 1
        if I == 1/2 and I3 == 1/2:
            return (B(0) * A(1/2) - np.sqrt(2) * B(1) * A(-1/2))/(np.sqrt(3))
        elif I == 1/2 and I3 == -1/2:
            return (B(0) * A(-1/2) - np.sqrt(2) * B(-1) * A(1/2))/(np.sqrt(3))
        elif I == 3/2 and I3 == 3/2:
            return B(1) * A(1/2)
        elif I == 3/2 and I3 == 1/2:
            return (B(1) * A(-1/2) + np.sqrt(2) * B(0) * A(1/2))/(np.sqrt(3))
        elif I == 3/2 and I3 == -1/2:
            return (B(-1) * A(1/2) + np.sqrt(2) * B(0) * A(-1/2))/(np.sqrt(3))
        elif I == 3/2 and I3 == -3/2:
            return B(-1) * A(-1/2)
            
    elif list(rep) == [1, 1]:
        # Equation 7.15: 1 ⊗ 1
        if I == 0 and I3 == 0:
            return (B(-1) * A(1) - B(0) * A(0) + B(1) * A(-1))/(np.sqrt(3))
        elif I == 1 and I3 == 1:
            return (B(0) * A(1) - B(1) * A(0))/(np.sqrt(2))
        elif I == 1 and I3 == 0:
            return (B(-1) * A(1) - B(1) * A(-1))/(np.sqrt(2))
        elif I == 1 and I3 == -1:
            return (B(-1) * A(0) - B(0) * A(-1))/(np.sqrt(2))
        elif I == 2 and I3 == 2:
            return B(1) * A(1)
        elif I == 2 and I3 == 1:
            return (B(0) * A(1) + B(1) * A(0))/(np.sqrt(2))
        elif I == 2 and I3 == 0:
            return (B(-1) * A(1) + 2 * B(0) * A(0) + B(1) * A(-1))/(np.sqrt(6))
        elif I == 2 and I3 == -1:
            return (B(-1) * A(0) + B(0) * A(-1))/(np.sqrt(2))
        elif I == 2 and I3 == -2:
            return B(-1) * A(-1)
            
    elif list(rep) == [1/2, 3/2]:
        # Equation 7.16: 1/2 ⊗ 3/2
        if I == 1 and I3 == 1:
            return (B(1/2) * A(1/2) - np.sqrt(3) * B(3/2) * A(-1/2))/2
        elif I == 1 and I3 == 0:
            return (B(-1/2) * A(1/2) - B(1/2) * A(-1/2))/np.sqrt(2)
        elif I == 1 and I3 == -1:
            return (np.sqrt(3) * B(-3/2) * A(1/2) - B(-1/2) * A(-1/2))/2
        elif I == 2 and I3 == 2:
            return B(3/2) * A(1/2)
        elif I == 2 and I3 == 1:
            return (B(-3/2) * A(1/2) + np.sqrt(3) * B(-1/2) * A(-1/2))/2
        elif I == 2 and I3 == 0:
            return (B(-1/2) * A(1/2) + B(1/2) * A(-1/2))/np.sqrt(2)
        elif I == 2 and I3 == -1:
            return (np.sqrt(3) * B(1/2) * A(1/2) + B(3/2) * A(-1/2))/2
        elif I == 2 and I3 == -2:
            return B(-3/2) * A(-1/2)
            
    elif list(rep) == [1, 3/2]:
        # Equation 7.17: 1 ⊗ 3/2
        if I == 1/2 and I3 == 1/2:
            return (B(-1/2) * A(1) - np.sqrt(2) * B(1/2) * A(0) + np.sqrt(3) * B(3/2) * A(-1))/(np.sqrt(6))
        elif I == 1/2 and I3 == -1/2:
            return (np.sqrt(3) * B(-3/2) * A(1) - np.sqrt(2) * B(-1/2) * A(0) + B(1/2) * A(-1))/(np.sqrt(6))
        elif I == 3/2 and I3 == 3/2:
            return (np.sqrt(2) * B(1/2) * A(1) - np.sqrt(3) * B(3/2) * A(0))/(np.sqrt(5))
        elif I == 3/2 and I3 == 1/2:
            return (4 * B(-1/2) * A(1) - np.sqrt(2) * B(1/2) * A(0) - 2 * np.sqrt(3) * B(3/2) * A(-1))/(np.sqrt(30))
        elif I == 3/2 and I3 == -1/2:
            return (2 * np.sqrt(3) * B(-3/2) * A(1) + np.sqrt(2) * B(-1/2) * A(0) - 4 * B(1/2) * A(-1))/(np.sqrt(30))
        elif I == 3/2 and I3 == -3/2:
            return (np.sqrt(3) * B(-3/2) * A(0) - np.sqrt(2) * B(-1/2) * A(-1))/(np.sqrt(5))
        elif I == 5/2 and I3 == 5/2:
            return B(3/2) * A(1)
        elif I == 5/2 and I3 == 3/2:
            return (np.sqrt(3) * B(1/2) * A(1) + np.sqrt(2) * B(3/2) * A(0))/(np.sqrt(5))
        elif I == 5/2 and I3 == 1/2:
            return (3 * B(-1/2) * A(1) + 3 * np.sqrt(2) * B(1/2) * A(0) + np.sqrt(3) * B(3/2) * A(-1))/(np.sqrt(30))
        elif I == 5/2 and I3 == -1/2:
            return (np.sqrt(3) * B(-3/2) * A(1) + 3 * np.sqrt(2) * B(-1/2) * A(0) + 3 * B(1/2) * A(-1))/(np.sqrt(30))
        elif I == 5/2 and I3 == -3/2:
            return (np.sqrt(2) * B(-3/2) * A(0) + np.sqrt(3) * B(-1/2) * A(-1))/(np.sqrt(5))
        elif I == 5/2 and I3 == -5/2:
            return B(-3/2) * A(-1)    
    else:
        raise ValueError(f"Representation {rep} not implemented")

    return None  # Return None if no matching I, I3 combination found
