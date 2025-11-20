from PyTorTractor import *
'''
we can handle following formats:
'<Corr>{J pion P=[1, -1, 0] A1um_1 SS_0} {pion P=[0, -3, 1] A1um_1 SS_0}</Corr>'
'<Corr>{pion P=[1, -1, 0] A1um_1 SS_0} {pion P=[0, -3, 1] A1um_1 SS_0}</Corr>'
'<Corr>{pion P=[1, -1, 0] A1um_1 SS_0} {pion P=[0, -3, 1] A1um_1 SS_0}</Corr>'
'<Corr>{isosinglet_kaon_kbar A1p_1 P=[0,0,0] A1u SS_0 P=[0,0,0] A2 SS_0} {isosinglet_kaon_kbar A1p_1 P=[0,0,0] A1u SS_0 P=[0,0,0] A2 SS_0}</Corr>'
'<Corr>{J isosinglet_kaon_kbar A1p_1 P=[0,0,0] A1u SS_0 P=[0,0,0] A2 SS_0} {isosinglet_kaon_kbar A1p_1 P=[0,0,0] A1u SS_0 P=[0,0,0] A2 SS_0}</Corr>'
'<Corr>{pion P=[1, -1, 0] A1um_1 SS_0} {J pion P=[0, -3, 1] A1um_1 SS_0} {pion P=[0, -3, 1] A1um_1 SS_0}</Corr>'
'''
def extract_wickpath(string):
    wickpath = ''
    for time_slice in string:
        splited_string = time_slice.split(' ')
        if splited_string[0] == 'J':
            wickpath += 'J'+splited_string[1] + '_'
        else:
            wickpath += splited_string[0] + '_'
    return wickpath[:-1]
def HCounter(string):
    return string.count('P=')
def PFiner_in1H(string):
    start = string.find('[') + 1
    end = string.find(']')
    p = string[start:end].split(',')
    px, py, pz = int(p[0]), int(p[1]), int(p[2])
    return (px, py, pz)
hdrn_types = {'pion': 'isovector_du',
              'eta': 'isoscalar', 'sigma': 'isoscalar', 'phi': 'isoscalar',
              'kaon': 'kaon_su', 'kbar': 'antikaon_ds',
              'delta': 'delta_uud','lambda': 'lambda_uds', 'nucleon': 'nucleon_uud',
              'omega': 'omega_sss', 'Sigma': 'sigma_uds', 'xi': 'xi_ssu'}
def is_meson(string):
    return string in ['pion', 'eta', 'sigma', 'phi' ,'kaon', 'kbar']
def O_path_finder(name_hadron, position):
    #if position[0] == 2:
    #    return '../Hadron_Info/current_operators.h5', 'meson_operators'
    meson_info_path  = '../Hadron_Info/meson_operators.h5'
    baryon_info_path = '../Hadron_Info/baryon_operators.h5'
    if is_meson(name_hadron):
        return meson_info_path, 'meson_operators'
    else:
        return baryon_info_path, 'baryon_operators'

def Two_path_finder(name_hadrons):
    MM_info_path  = '../Hadron_Info/meson_meson_operators.h5'
    MB_info_path  = '../Hadron_Info/meson_baryon_operators.h5'
    BB_info_path  = '../Hadron_Info/baryon_baryon_operators.h5'
    if name_hadrons == ['meson_operators', 'meson_operators']:
        return MM_info_path
    elif name_hadrons == ['meson_operators', 'baryon_operators']:
        return MB_info_path
    elif name_hadrons == ['baryon_operators', 'baryon_operators']:
        return BB_info_path
    else:
        raise ValueError('Error 10')

def OneHdrn_initializer(string, position, dlen = None):
    split_info = string.split(' ')
    Hname = split_info[0]
    if Hname == 'J':
        Hname = split_info[1]
        position = (2, position[1])
    H_path_info = O_path_finder(Hname, position)
    htype = hdrn_types[Hname]
    Disp_info = split_info[-1]
    Group_info = split_info[-2]
    momentum = PFiner_in1H(string)
    if H_path_info[1] == 'meson_operators' and dlen is not None:
        dlen = None
        print('Here: ', H_path_info[1])
        print('dlen =', dlen)
    #return 
    #print(f'Hadron(File_Info_Path = {H_path_info[0]}, Hadron_Type = {H_path_info[1]},Hadron_Position = {position},
    #Flavor = {htype}, Momentum = {momentum}, LGIrrep #= {Group_info}, Displacement = {Disp_info})')
    return Hadron(File_Info_Path = H_path_info[0], Hadron_Type = H_path_info[1], Hadron_Position = position,
                  Flavor = htype, Momentum = momentum, LGIrrep = Group_info, Displacement = Disp_info, dlen = dlen), H_path_info[1], momentum
def OneTDecomposer(string = None, dlen = None, position = None, Two_Hadron_Momentum = None, strangeness = None, OpNum = 0):
    if HCounter(string) == 1:#One Hadron on a Time Slice
        return OneHdrn_initializer(string, position, dlen = dlen)[0]
    elif HCounter(string) == 2:#Two Hadrons on a Time Slice
        split_1 = string.split(' ')
        if len(split_1) == 9:
            split_1 = string[2:].split(' ')
            position = (2, position[1])
        split_2 = split_1[0].split('_')
        H1_String = split_2[1] + ' ' + split_1[2] + ' ' + split_1[3] + ' ' + split_1[4]
        H1_Position = (position[0], position[1])
        H2_String = split_2[2] + ' ' + split_1[5] + ' ' + split_1[6] + ' ' + split_1[7]
        H2_Position = (position[0], position[1]+1)
        if dlen is None:
            dlen = [None, None]
        elif isinstance(dlen, str):
            dlen = [dlen, dlen]
        H1 = OneHdrn_initializer(H1_String, H1_Position, dlen = dlen[0])
        H2 = OneHdrn_initializer(H2_String, H2_Position, dlen = dlen[1])
        Hadron1 = H1[0]
        Hadron2 = H2[0]
        overall_Group = split_1[1]
        overall_path = Two_path_finder([H1[1], H2[1]])
        if Two_Hadron_Momentum is None:
            Two_Hadron_Momentum = tuple([H1[2][i] + H2[2][i] for i in range(3)])
        return TwoHadron(File_Info_Path = overall_path, Total_Momentum = Two_Hadron_Momentum, LGIrrep = overall_Group,
                         Hadron1 = Hadron1, Hadron2 = Hadron2, OpNum = OpNum, strangeness = strangeness)
    else:
        raise ValueError('Generalize the string-method to three Hadrons')
class Hadrons_from_Strings:
    def __init__(self, string = None, dlen = None, dlen_left = None, dlen_right = None, overallMom = None, strangeness = None, OpNum = 0):
        self.string = string[6:-7].split('{')[1:]
        self.overallMom = overallMom
        self.strangeness = strangeness
        self.OpNum = OpNum
        self.dlen = dlen
        self.dlen_left = dlen_left
        self.dlen_right = dlen_right
        self.N = len(self.string)
    def prepare_hadrons(self):        
        if self.N == 1:
            all_info = self.string[0][:-1]
            wickpath = extract_wickpath([all_info])
            return [OneTDecomposer(string = all_info, dlen = self.dlen, position = (0,0), Two_Hadron_Momentum = self.overallMom,
                                  strangeness = self.strangeness, OpNum = self.OpNum)], wickpath
        elif self.N == 2:
            left_Info = self.string[0][:-2]
            right_Info = self.string[1][:-1]
            wickpath = extract_wickpath([left_Info, right_Info])
            dlens1 = self.dlen_left
            dlens2 = self.dlen_right
            if (self.dlen_right is None) and (self.dlen_left is None):
                dlens1 = self.dlen
                dlens2 = self.dlen
            left_hadrons = OneTDecomposer(string = left_Info, dlen = dlens1, position = (1,0), Two_Hadron_Momentum = self.overallMom,
                                  strangeness = self.strangeness, OpNum = self.OpNum)
            right_hadrons = OneTDecomposer(string = right_Info, dlen = dlens2, position = (0,0), Two_Hadron_Momentum = self.overallMom,
                                  strangeness = self.strangeness, OpNum = self.OpNum)
            return [left_hadrons, right_hadrons], wickpath
        elif self.N == 3:
            sink_Info = self.string[0][:-2]
            current_Info = self.string[1][:-2]
            source_Info = self.string[2][:-1]
            wickpath = extract_wickpath([sink_Info, current_Info, source_Info])
            sink_hadrons = OneTDecomposer(string = sink_Info, position = (1,0), Two_Hadron_Momentum = self.overallMom,
                                  strangeness = self.strangeness, OpNum = self.OpNum)
            current_hadrons = OneTDecomposer(string = current_Info, position = (2,0), Two_Hadron_Momentum = self.overallMom,
                                  strangeness = self.strangeness, OpNum = self.OpNum)
            source_hadrons = OneTDecomposer(string = source_Info, position = (0,0), Two_Hadron_Momentum = self.overallMom,
                                  strangeness = self.strangeness, OpNum = self.OpNum)
            return [sink_hadrons, current_hadrons, source_hadrons], wickpath
        else:
            raise ValueError('Update the reader methods')
    def get_Hadrons(self):
        return self.prepare_hadrons()[0]
    def get_Wickpath(self):
        return self.prepare_hadrons()[1]
'''
class P_LittleGroup:
    def __init__(self, p_list):
        self.p_array = np.array(p_list)
        self.zr      = np.array([1 for i in p_list if i == 0]).sum()
        self.max     = self.p_array.max()
        self.maxoft  = np.array([1 for i in p_list if i == self.max]).sum()
        ue           = list(set(p_list))
        self.ue      = set((np.array([i for i in ue if i != 0])/self.max).tolist())
    def mom_type(self):
        if self.zr == 3:
            return 'rest_frame'
        elif self.zr == 2:
            return 'on_axis'
        elif self.zr == 1:
            if len(self.ue) == 1:
                return 'planar-diagonal'
            else:
                if self.ue == {1,2}:
                    return 'Cs_for_(0,1,2)'
                else:
                    return 'Unknown-type'
                    #raise ValueError(f'Failed to identity the little group for {self.p_list}')
        else:
            if len(self.ue) == 1:
                return 'cubic-diagonal'
            elif len(self.ue) == 2:
                if self.ue == {0.5,1}:
                    if self.maxoft == 1:
                        return 'Cs_for_(1,1,2)'
                    elif self.maxoft == 2:
                        return 'Cs_for_(1,2,2)'
                elif self.ue == {}
                else:
                    return 'Unknown-type'
                    #raise ValueError(f'Failed to identity the little group for {self.p_list}')
            else:
                return 'Unknown-type'
                #raise ValueError(f'Failed to identity the little group for {self.p_list}')

'''

single_meson_correlators = [
    '<Corr>{pion P=[0, 0, 0] A1um_1 SS_0} {pion P=[0, 0, 0] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[0, 0, 1] A1um_1 SS_1} {pion P=[0, 0, 1] A1um_1 SS_1}</Corr>',
    '<Corr>{pion P=[0, 1, 0] A1um_1 SS_1} {pion P=[0, 1, 0] A1um_1 SS_1}</Corr>',
    '<Corr>{pion P=[1, 0, 0] A1um_1 SS_1} {pion P=[1, 0, 0] A1um_1 SS_1}</Corr>',
    '<Corr>{pion P=[0, 1, 1] A1um_1 SS_0} {pion P=[0, 1, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 0, 1] A1um_1 SS_0} {pion P=[1, 0, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 1, 0] A1um_1 SS_0} {pion P=[1, 1, 0] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 1, 1] A1um_1 SS_0} {pion P=[1, 1, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[0, 0, 2] A1um_1 SS_1} {pion P=[0, 0, 2] A1um_1 SS_1}</Corr>',
    '<Corr>{pion P=[0, 2, 0] A1um_1 SS_1} {pion P=[0, 2, 0] A1um_1 SS_1}</Corr>',
    '<Corr>{pion P=[2, 0, 0] A1um_1 SS_1} {pion P=[2, 0, 0] A1um_1 SS_1}</Corr>',
    '<Corr>{pion P=[0, 1, 2] A1um_1 SS_0} {pion P=[0, 1, 2] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[0, 2, 1] A1um_1 SS_0} {pion P=[0, 2, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 0, 2] A1um_1 SS_0} {pion P=[1, 0, 2] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 2, 0] A1um_1 SS_0} {pion P=[1, 2, 0] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[2, 0, 1] A1um_1 SS_0} {pion P=[2, 0, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[2, 1, 0] A1um_1 SS_0} {pion P=[2, 1, 0] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 1, 2] A1um_1 SS_0} {pion P=[1, 1, 2] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 2, 1] A1um_1 SS_0} {pion P=[1, 2, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[2, 1, 1] A1um_1 SS_0} {pion P=[2, 1, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[0, -1, 1] A1um_1 SS_0} {pion P=[0, -1, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-1, 0, 1] A1um_1 SS_0} {pion P=[-1, 0, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-1, 1, 0] A1um_1 SS_0} {pion P=[-1, 1, 0] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-1, 1, 1] A1um_1 SS_0} {pion P=[-1, 1, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, -1, 1] A1um_1 SS_0} {pion P=[1, -1, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 1, -1] A1um_1 SS_0} {pion P=[1, 1, -1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[0, -1, 2] A1um_1 SS_0} {pion P=[0, -1, 2] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[0, -2, 1] A1um_1 SS_0} {pion P=[0, -2, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-1, 0, 2] A1um_1 SS_0} {pion P=[-1, 0, 2] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-1, 2, 0] A1um_1 SS_0} {pion P=[-1, 2, 0] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-2, 0, 1] A1um_1 SS_0} {pion P=[-2, 0, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-2, 1, 0] A1um_1 SS_0} {pion P=[-2, 1, 0] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-1, 1, 2] A1um_1 SS_0} {pion P=[-1, 1, 2] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, 1, -2] A1um_1 SS_0} {pion P=[1, 1, -2] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-1, 2, 1] A1um_1 SS_0} {pion P=[-1, 2, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[1, -2, 1] A1um_1 SS_0} {pion P=[1, -2, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[-2, 1, 1] A1um_1 SS_0} {pion P=[-2, 1, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{pion P=[2, -1, 1] A1um_1 SS_0} {pion P=[2, -1, 1] A1um_1 SS_0}</Corr>',
    '<Corr>{kaon P=[0, 0, 0] A1u_1 SS_0} {kaon P=[0, 0, 0] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[0, 0, 1] A1u_1 SS_1} {kaon P=[0, 0, 1] A1u_1 SS_1}</Corr>',
    '<Corr>{kaon P=[0, 1, 0] A1u_1 SS_1} {kaon P=[0, 1, 0] A1u_1 SS_1}</Corr>',
    '<Corr>{kaon P=[1, 0, 0] A1u_1 SS_1} {kaon P=[1, 0, 0] A1u_1 SS_1}</Corr>',
    '<Corr>{kaon P=[0, 1, 1] A1u_1 SS_0} {kaon P=[0, 1, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 0, 1] A1u_1 SS_0} {kaon P=[1, 0, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 1, 0] A1u_1 SS_0} {kaon P=[1, 1, 0] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 1, 1] A1u_1 SS_0} {kaon P=[1, 1, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[0, 0, 2] A1u_1 SS_1} {kaon P=[0, 0, 2] A1u_1 SS_1}</Corr>',
    '<Corr>{kaon P=[0, 2, 0] A1u_1 SS_1} {kaon P=[0, 2, 0] A1u_1 SS_1}</Corr>',
    '<Corr>{kaon P=[2, 0, 0] A1u_1 SS_1} {kaon P=[2, 0, 0] A1u_1 SS_1}</Corr>',
    '<Corr>{kaon P=[0, 1, 2] A1u_1 SS_0} {kaon P=[0, 1, 2] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[0, 2, 1] A1u_1 SS_0} {kaon P=[0, 2, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 0, 2] A1u_1 SS_0} {kaon P=[1, 0, 2] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 2, 0] A1u_1 SS_0} {kaon P=[1, 2, 0] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[2, 0, 1] A1u_1 SS_0} {kaon P=[2, 0, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[2, 1, 0] A1u_1 SS_0} {kaon P=[2, 1, 0] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 1, 2] A1u_1 SS_0} {kaon P=[1, 1, 2] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 2, 1] A1u_1 SS_0} {kaon P=[1, 2, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[2, 1, 1] A1u_1 SS_0} {kaon P=[2, 1, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[0, -1, 1] A1u_1 SS_0} {kaon P=[0, -1, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-1, 0, 1] A1u_1 SS_0} {kaon P=[-1, 0, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-1, 1, 0] A1u_1 SS_0} {kaon P=[-1, 1, 0] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-1, 1, 1] A1u_1 SS_0} {kaon P=[-1, 1, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, -1, 1] A1u_1 SS_0} {kaon P=[1, -1, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 1, -1] A1u_1 SS_0} {kaon P=[1, 1, -1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[0, -1, 2] A1u_1 SS_0} {kaon P=[0, -1, 2] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[0, -2, 1] A1u_1 SS_0} {kaon P=[0, -2, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-1, 0, 2] A1u_1 SS_0} {kaon P=[-1, 0, 2] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-1, 2, 0] A1u_1 SS_0} {kaon P=[-1, 2, 0] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-2, 0, 1] A1u_1 SS_0} {kaon P=[-2, 0, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-2, 1, 0] A1u_1 SS_0} {kaon P=[-2, 1, 0] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-1, 1, 2] A1u_1 SS_0} {kaon P=[-1, 1, 2] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, 1, -2] A1u_1 SS_0} {kaon P=[1, 1, -2] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-1, 2, 1] A1u_1 SS_0} {kaon P=[-1, 2, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[1, -2, 1] A1u_1 SS_0} {kaon P=[1, -2, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[-2, 1, 1] A1u_1 SS_0} {kaon P=[-2, 1, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{kaon P=[2, -1, 1] A1u_1 SS_0} {kaon P=[2, -1, 1] A1u_1 SS_0}</Corr>',
    '<Corr>{eta P=[0, 0, 0] A1up_1 SS_0} {eta P=[0, 0, 0] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[0, 0, 1] A1up_1 SS_1} {eta P=[0, 0, 1] A1up_1 SS_1}</Corr>',
    '<Corr>{eta P=[0, 1, 0] A1up_1 SS_1} {eta P=[0, 1, 0] A1up_1 SS_1}</Corr>',
    '<Corr>{eta P=[1, 0, 0] A1up_1 SS_1} {eta P=[1, 0, 0] A1up_1 SS_1}</Corr>',
    '<Corr>{eta P=[0, 1, 1] A1up_1 SS_0} {eta P=[0, 1, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 0, 1] A1up_1 SS_0} {eta P=[1, 0, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 1, 0] A1up_1 SS_0} {eta P=[1, 1, 0] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 1, 1] A1up_1 SS_0} {eta P=[1, 1, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[0, 0, 2] A1up_1 SS_1} {eta P=[0, 0, 2] A1up_1 SS_1}</Corr>',
    '<Corr>{eta P=[0, 2, 0] A1up_1 SS_1} {eta P=[0, 2, 0] A1up_1 SS_1}</Corr>',
    '<Corr>{eta P=[2, 0, 0] A1up_1 SS_1} {eta P=[2, 0, 0] A1up_1 SS_1}</Corr>',
    '<Corr>{eta P=[0, 1, 2] A1up_1 SS_0} {eta P=[0, 1, 2] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[0, 2, 1] A1up_1 SS_0} {eta P=[0, 2, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 0, 2] A1up_1 SS_0} {eta P=[1, 0, 2] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 2, 0] A1up_1 SS_0} {eta P=[1, 2, 0] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[2, 0, 1] A1up_1 SS_0} {eta P=[2, 0, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[2, 1, 0] A1up_1 SS_0} {eta P=[2, 1, 0] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 1, 2] A1up_1 SS_0} {eta P=[1, 1, 2] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 2, 1] A1up_1 SS_0} {eta P=[1, 2, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[2, 1, 1] A1up_1 SS_0} {eta P=[2, 1, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[0, -1, 1] A1up_1 SS_0} {eta P=[0, -1, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-1, 0, 1] A1up_1 SS_0} {eta P=[-1, 0, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-1, 1, 0] A1up_1 SS_0} {eta P=[-1, 1, 0] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-1, 1, 1] A1up_1 SS_0} {eta P=[-1, 1, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, -1, 1] A1up_1 SS_0} {eta P=[1, -1, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 1, -1] A1up_1 SS_0} {eta P=[1, 1, -1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[0, -1, 2] A1up_1 SS_0} {eta P=[0, -1, 2] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[0, -2, 1] A1up_1 SS_0} {eta P=[0, -2, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-1, 0, 2] A1up_1 SS_0} {eta P=[-1, 0, 2] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-1, 2, 0] A1up_1 SS_0} {eta P=[-1, 2, 0] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-2, 0, 1] A1up_1 SS_0} {eta P=[-2, 0, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-2, 1, 0] A1up_1 SS_0} {eta P=[-2, 1, 0] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-1, 1, 2] A1up_1 SS_0} {eta P=[-1, 1, 2] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, 1, -2] A1up_1 SS_0} {eta P=[1, 1, -2] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-1, 2, 1] A1up_1 SS_0} {eta P=[-1, 2, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[1, -2, 1] A1up_1 SS_0} {eta P=[1, -2, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[-2, 1, 1] A1up_1 SS_0} {eta P=[-2, 1, 1] A1up_1 SS_0}</Corr>',
    '<Corr>{eta P=[2, -1, 1] A1up_1 SS_0} {eta P=[2, -1, 1] A1up_1 SS_0}</Corr>']