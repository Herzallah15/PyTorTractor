# system modules
import os
import sys

# optional: falls psutil noch gebraucht wird
#import psutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Source")))
# project modules
from contractions_handler import *
from Hadron_Info_Converter import *
from Hadrontractions_Converter import *
from PyTorDefinitions import *
from PyTor_S_Definitions import *
from PyTorTractor_SingleHadron import *
from PyTorTractor import *
from WickDiagrams_Merger import *
import time

input_paths = []


#<Corr>{J pion P=[0,0,0] A1um_1 SS_0} {pion P=[0,0,0] A1um_1 SS_0}</Corr>
O1 = Pion(1)
O1_ontime = OpTimeSlice(2, O1)
O2 = bar(Pion(1))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath =  '../WickDiagrams/Jpion_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path =Dpath)


#<Corr>{J kaon P=[0,0,0] T1u_3 SS_0} {isodoublet_kaon_pion T1u_3 P=[0,0,1] A2 SS_1 P=[0,0,-1] A2m SS_1}</Corr>
O1 = Kaon(1/2)
O1_ontime = OpTimeSlice(2, O1)
O2 = bar(twoHO(rep=(1/2,1), I=1/2, I3=1/2, A=Kaon, B=Pion))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/Jkaon_isodoublet_kaon_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)


#<Corr> {phi P=[0,0,0] A1gp_1 SS_0} {phi P=[0,0,0] A1gp_1 SS_0}</Corr>
O1 = Phi()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Phi())
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/phi_phi.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)

#<Corr> {eta P=[0,0,0] A1gp_1 SS_0} {eta P=[0,0,0] A1gp_1 SS_0}</Corr>
O1 = Eta()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Eta())
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/eta_eta.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)


#<Corr>{eta P=(0,0,0) A1gp_1 SS_0} {phi P=(0,0,0) A1gp_1 SS_0}</Corr> 
O1 = Eta()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Phi())
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/eta_phi.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#

    
#<Corr>{eta P=[0,0,0] A1gp_1 SS_0} {isosinglet_kaon_kbar A1gp_1 [P=[0,0,0] A1u SS_0] [P=[0,0,0] A1u SD_1]}</Corr>
O1 = Eta()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1/2,1/2), I=0, I3=0, A=Kaon, B=KaonC))#bar((Kaon(1/2) * KaonC(-1/2) - Kaon(-1/2) * KaonC(1/2))/(np.sqrt(2)))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/eta_isosinglet_kaon_kbar.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#

#<Corr>{phi P=[0,0,0] A1gp_1 SS_0} {isosinglet_kaon_kbar A1gp_1 [P=[0,0,0] A1u SS_0] [P=[0,0,0] A1u SD_1]}</Corr>
O1 = Phi()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1/2,1/2), I=0, I3=0, A=Kaon, B=KaonC))#bar((Kaon(1/2) * KaonC(-1/2) - Kaon(-1/2) * KaonC(1/2))/(np.sqrt(2)))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/phi_isosinglet_kaon_kbar.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#



#<Corr>{isosinglet_kaon_kbar A1gp_1 [P=[0,0,0] A1u SS_0] [P=[0,0,0] A1u SS_0]} {isosinglet_kaon_kbar A1gp_1 [P=[0,0,0] A1u SS_0] [P=[0,0,0] A1u SD_1]}</Corr>
O1         = twoHO(rep=(1/2,1/2), I=0, I3=0, A=Kaon, B=KaonC)#(Kaon(1/2) * KaonC(-1/2) - Kaon(-1/2) * KaonC(1/2))/(np.sqrt(2))
O1_ontime  = OpTimeSlice(1, O1)
O2         = bar(O1)
O2_ontime  = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result     = Correlator.Laudtracto()
Dpath = '../WickDiagrams/isosinglet_kaon_kbar_isosinglet_kaon_kbar.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#


#<Corr>{isodoublet_kaon_pion T1u_3 [P=[0,0,1] A2 SS_1] [P=[0,0,-1] A2m SS_1]} {isodoublet_kaon_pion T1u_3 [P=[0,0,1] A2 SS_0] [P=[0,0,-1] A2m SS_1]}</Corr>
O1 = twoHO(rep=(1/2,1), I=1/2, I3=1/2, A=Kaon, B=Pion)#(Kaon(1/2) * Pion(0) - np.sqrt(2) * Kaon(-1/2) * Pion(1))/(np.sqrt(3))
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/isodoublet_kaon_pion_isodoublet_kaon_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#

#<Corr>{isodoublet_kaon_pion T1u_3 [P=[0,0,1] A2 SS_0] [P=[0,0,-1] A2m SS_1]} {kaon P=[0,0,0] T1u_3 SS_0}</Corr>
O1 = twoHO(rep=(1/2,1), I=1/2, I3=1/2, A=Kaon, B=Pion)#(Kaon(1/2) * Pion(0) - (2**0.5) * Kaon(-1/2) * Pion(1))/(3**0.5)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Kaon(1/2))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/isodoublet_kaon_pion_kaon.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#


#<Corr>{kaon P=[0,0,0] T1u_3 SS_0} {isodoublet_kaon_pion T1u_3 [P=[0,0,1] A2 SS_1] [P=[0,0,-1] A2m SS_1]}</Corr>
O1 = Kaon(1/2)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1/2,1), I=1/2, I3=1/2, A=Kaon, B=Pion))#bar( (Kaon(1/2) * Pion(0) - (2**0.5) * Kaon(-1/2) * Pion(1))/(3**0.5))

O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/kaon_isodoublet_kaon_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#



#<Corr>{pion P=[0,0,0] T1up_3 SS_0} {isotriplet_pion_pion T1up_3 P=[0,0,1] A2m SS_1 P=[0,0,-1] A2m SS_1}</Corr>
O1 = Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = twoHO(rep=(1,1), I=1, I3=1, A=Pion, B=Pion)#TwoHadronAnnihilation(rep=(1,1), I=1, I3=1, A=Pion, B=Pion)
O2 = bar(O2)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/pion_isotriplet_pion_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#



#<Corr>{isoquartet_kaon_pion A1g_1 P=[0,0,0] A1u SS_0 P=[0,0,0] A1um SS_0} {isoquartet_kaon_pion A1g_1 P=[0,0,0] A1u SS_0 P=[0,0,0] A1um SS_0}</Corr>
O1 = twoHO(rep=(1/2,1), I=3/2, I3=3/2, A=Kaon, B=Pion)#Kaon(1/2) * Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/isoquartet_kaon_pion_isoquartet_kaon_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#



#<Corr>{lambda P=[0,0,0] G1u_1 SS_0} {isosinglet_kbar_nucleon G1u_1 P=[0,0,0] A1u SS_0 P=[0,0,0] G1g SS_0}</Corr>
#I = 0, I_3 = 0
O1 = Lambda()
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1/2,1/2), I=0, I3=0, A=KaonC, B=Nucleon))#(bar(Nucleon(1/2)) * bar(KaonC(-1/2)) - bar(Nucleon(-1/2)) * bar(KaonC(1/2)))/(2**(1/2))
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/lambda_isosinglet_kbar_nucleon.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#



#<Corr>{isosinglet_kbar_nucleon G1u_1 P=[0,0,0] A1u SS_0 P=[0,0,0] G1g SS_0} {lambda P=[0,0,0] G1u_1 SS_0}</Corr>
#I = 0, I_3 = 0
O1 = twoHO(rep=(1/2,1/2), I=0, I3=0, A=KaonC, B=Nucleon)#( KaonC(-1/2) * Nucleon(1/2) - KaonC(1/2) * Nucleon(-1/2) )/(2**(1/2))
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(Lambda())
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/isosinglet_kbar_nucleon_lambda.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#

#<Corr>{isotriplet_pion_pion T1up_3 [P=[0,0,1] A2m SS_1] [P=[0,0,-1] A2m SS_1]} {isotriplet_pion_pion T1up_3 [P=[0,0,1] A2m SS_1] [P=[0,0,-1] A2m SS_1]}</Corr>
O1 = twoHO(rep=(1,1), I=1, I3=1, A=Pion, B=Pion)#(Pion(0) * Pion(1) - Pion(1) * Pion(0))/(2**(0.5))
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/isotriplet_pion_pion_isotriplet_pion_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)


#<Corr>{pion P=[0,0,0] T1up_3 SS_0} {isotriplet_pion_pion T1up_3 P=[0,0,1] A2m SS_1 P=[0,0,-1] A2m SS_1}</Corr>
O1 = Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(twoHO(rep=(1,1), I=1, I3=1, A=Pion, B=Pion))#bar(Pion(0) * Pion(1) - Pion(1) * Pion(0)) / (2**0.5)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/pion_isotriplet_pion_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#


#<Corr>{kaon P=[0,0,0] A1u_1 SS_0} {kaon P=[0,0,0] A1u_1 SS_0}</Corr>
O1 = Kaon(1/2)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/kaon_kaon.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#



#<Corr>{lambda P=[0,0,1] G2_1 SS_0} {lambda P=[0,0,1] G2_1 SS_0}</Corr>
O1 = Lambda(0)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/lambda_lambda.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#

#<Corr>{pion P=[0,0,0] A1um_1 SS_0} {pion P=[0,0,0] A1um_1 SS_0}</Corr>
O1 = Pion(1)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/pion_pion.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#


#<Corr>{sigma P=[0,0,0] A1gp_1 SS_0} {sigma P=[0,0,0] A1gp_1 SS_0}</Corr>
O1 = sigma(0)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/sigma_sigma.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
#



#<Corr>{nucleon P=[0,0,0] G1g_1 SS_0} {nucleon P=[0,0,0] G1g_1 SS_0}</Corr> 
O1 = Nucleon(1/2)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
Dpath = '../WickDiagrams/nucleon_nucleon.hdf5'
input_paths.append(Dpath)
writeresults(Result, O1_ontime, O2_ontime, path = Dpath)
print(len(input_paths))
input_paths = list(set(input_paths))
print(len(input_paths))
merge_hdf5_files(output_path='../WickDiagrams/Basic_Tests.hdf5', input_paths=input_paths)