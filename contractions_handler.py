from itertools import combinations
# a class to generate quarks that remember from  which  hadron  they   are
# ff =           the numerical factor, with which the hadron is multiplied
# hdrn_t        =   the  hadron   type,   in   which   the   quark   lives
# mntm          =           the      momentum      of      the      hadron
# flvr          =           the        flavor        of     the      quark
# qrk_hdrn_p    =      the    quark    position    inside    the    hadron
class qrk:
    def __init__(self, hadron, ff, flvr, qrk_hdrn_p):
        self.hadron     = hadron
        self.ff         = ff
        self.hadron     = hadron
        self.hdrn_t     = self.hadron.ghdrn_t()
        self.mntm       = self.hadron.gmntm()
        self.flvr       = flvr
        self.qrk_hdrn_p = qrk_hdrn_p
    def ghadron():
        return self.hadron
    def gff(self):
        return self.ff
    def ghdrn_t(self):
        return self.hdrn_t
    def gmntm(self):
        return self.mntm
    def gflvr(self):
        return self.flvr
    def gqrk_hdrn_p(self):
        return self.qrk_hdrn_p
    def light_quarks_finder(self):
        if self.flvr == 'u' or self.flvr == 'd':
            return 'Y'
        elif self.flvr == 'uB' or self.flvr == 'dB':
            return 'YB'
        else:
            return self.flvr
    def gqrkprp(self):
        return (self.hdrn_t, self.mntm, self.light_quarks_finder(), self.qrk_hdrn_p)
    def __eq__(self, other):
        if isinstance(other, qrk):
            return self.gqrkprp() == other.gqrkprp()




# a class to generate one single-hadron state, that  has no  superposition
# this will be the most important calss, from which all other classes with
# multihadrons      and      superpositions      will     be     generated
# as shown below all properties of the hadron are inherited by the  quarks
# hdrn_t corresponds to the type of the hadron. It must end with B in case
# the     hadron     is         taken     to         be     in         bar
# ff corresponds to the numerical factor multiplied by the qurak structure
# mntm    is     the      momentum       carried       by    the    hadron
# qrks corresponds to the flavor structure of the  hadron,  they  must  be
# given  in   the   correct   order,   e.g.,    for   Nucleonp   we   have
# flavor structure is uud, and hence, qrks must be given as: 'u', 'u', 'd'
# barness   will     be     used   as  True in   the  bar  fuction,  which
# creates          the          corresponding    bar-version of the hadron
# O is a functiom to generate a list of all quarks (qrk-type) inside of it
# it     can    take     an argument "ffex" if we need the quark structure
# multiplied     by an  extra  factor, as required for multi-hadron states
# like h2 and h3s. When  the  hdrn-type     object     is    called    via 
# bar,     the     block     under  "if barness"         is      evaluated
# otherwise, the block under "if not barness" will be executed accordingly
# the numerical factor of the hadron is forwarded to only one        quark
# all     other     quarks     obtain    a    numerical  factor    of    1
# we                     overload             operators                 to
# add 2 single hadrons to each  others    to    form    a    superposition
# add  one   single   hadron   to  a   superposition   of  single  hadrons
# multiply     a       single       hadron       with       a       number
# multiply two single hadrons with each others to build  a  rank-2  tensor
# multiply one single hadron with a superposition of     single    hadrons
# multiply one single hadron  with   rank-2  tensor  (i.e. a h2-object) to
# build  a     rank-3     tensor    i.e.,    a   three-hadron state object
# of          the         type              h3           ( three Hadrons )
# divide     a       single        hadron        with       a       number
class hdrn:
    def __init__(self, ff, hdrn_t, mntm, *qrks, barness = False):
        self.ff      = ff
        self.hdrn_t  = hdrn_t
        self.mntm    = mntm
        if len(qrks) > 0 and isinstance(qrks[-1], bool):
            self.qrks = qrks[:-1]
            self.barness = qrks[-1]
        else:
            self.qrks = qrks
            self.barness = barness
    def gff(self):
        return self.ff
    def ghdrn_t(self):
        return self.hdrn_t
    def gmntm(self):
        return self.mntm
    def gqrks(self):
        return list(self.qrks)
    def gbarness(self):
        return self.barness
    def gallprp(self):
        return [self.hdrn_t, self.mntm, list(self.qrks), self.barness]
    def gAhdrn(self):
        return [self]
    def O(self, ffex = None):
        operator     = []
        nmrclfctr    = [1 for i in self.gqrks()]
        if ffex == None:
            nmrclfctr[0] = self.gff()
        else:
            nmrclfctr[0] = self.gff() * ffex
        if not self.barness:
            for i, qr in enumerate(self.qrks):
                operator.append(qrk(self, nmrclfctr[i], qr, i))
            return operator
        elif self.barness:
            karm = len(self.qrks) - 1
            for i, qr in enumerate(self.qrks):
                operator.append(qrk(self, nmrclfctr[i], qr, karm - i))
            return operator
        else:
            raise TypeError(f"Error in generating the list containing the flavor structure of the hadron")
    def __eq__(self, other):
        return isinstance(other, hdrn) and self.gallprp() == other.gallprp()
    def __add__(self, other):
        if isinstance(other, hdrn):
            if self == other:
                nff = self.gff() + other.gff()
                return hdrn(nff, self.hdrn_t, self.mntm, *self.qrks, barness = self.barness)
            else:
                return mhdrn(2, self, other)
        elif isinstance(other, mhdrn):
            nwmhdrn    = []
            appearence = 0
            for i, hadron in enumerate(other.gAhdrn()):
                if (hadron == self) and (appearence == 0):
                    nhdrn = hadron + self
                    nwmhdrn.append(nhdrn)
                    appearence += 1
                else:
                    nwmhdrn.append(hadron)
            if appearence == 0:
                nwlngt  = other.glngt()  + 1
                nwmhdrn.append(self)
                return mhdrn(nwlngt, *nwmhdrn)
            else:
                nwlngt  = other.glngt()
                return mhdrn(nwlngt, *nwmhdrn)
    def __radd__(self, other):
        if isinstance(other, mhdrn):
            return self + other
        else:
            raise TypeError(f"Unsupported operand type(s) for +")
    def __sub__(self, other):
        if isinstance(other, hdrn):
            return self + (-1 * other)
        elif isinstance(other, mhdrn):
            return self + (-1 * other)
    def __rsub__(self, other):
        if isinstance(other, hdrn):
            return other + (-1 * self)
        elif isinstance(other, mhdrn):
            return other + (-1 * self)
        else:
            raise TypeError(f"Unsupported operand type(s) for -")

    def __mul__(self, other):
        if isinstance(other, (int, float)):#Include sp.Basic, sp.Number
            nff = other * (self.gff())
            return hdrn(nff, self.hdrn_t, self.mntm, *self.qrks, barness=self.barness)
        elif isinstance(other, hdrn):
            return h2(self, other)
        elif isinstance(other, mhdrn):
            nh2 = []
            for hadron in other.gAhdrn():
                nh2.append(self * hadron)
            return mh2(other.glngt(), *nh2)
        elif isinstance(other, h2):
            hdrnpair = h2(self, other.ghdrn1())
            return h3(hdrnpair, other.ghdrn2())
        elif isinstance(other, mh2):
            nh3 = []
            for two_hadrons in other.gAhdrn():
                nh3.append(self * two_hadrons)
            return mh3(other.glngt(), *nh3)
        elif isinstance(other, h3):
            return h4(self * other.ghdrn1() * other.ghdrn2(), other.ghdrn3())
        elif isinstance(other, mh3):
            nh4 = []
            for triplet_hadrons in other.gAhdrn():
                nh4.append(self * triplet_hadrons)
            return mh4(self.glngt(), *nh4)
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, mhdrn):
            nh2 = []
            for hadron in other.gAhdrn():
                nh2.append(hadron * self)
            return mh2(other.glngt(), *nh2)
        elif isinstance(other, h2):
            return h3(other, self)
        elif isinstance(other, mh2):
            nh3 = []
            for two_hadrons in other.gAhdrn():
                nh3.append(two_hadrons * self)
            return mh3(other.glngt(), *nh3)
        elif isinstance(other, h3):
            return h4(other, self)
        elif isinstance(other, mh3):
            nh4 = []
            for triplet_hadrons in other.gAhdrn():
                nh4.append(triplet_hadrons * self)
            return mh4(self.glngt(), *nh4)
        raise TypeError(f"Unsupported operand type(s) for *")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            nff = (self.gff())/other
            return hdrn(nff, self.hdrn_t, self.mntm, *self.qrks, barness=self.barness)
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self / other
        else:
            raise TypeError(f"Unsupported operand type(s) for /")



# function     to     generate    the   bar version of a one single-hadron
# if the hadron is to be taken in bar, then its name will    be    changed
# B is removed  when  taking  the  bar of a hadron which is already in bar
# or  B is   added    in  when     the    hadron    is    not    in    bar
def barOH(hdrn_state):

    ff     = hdrn_state.gff()
    mntm   = hdrn_state.gmntm()
    qrks   = hdrn_state.gqrks()

    if hdrn_state.ghdrn_t()[-1] == 'B':
        print("You are taking the bar of a hadron already in bar!\n")
        print("Notice that quark positions in a bar-bar hadron follow the order")
        print("of the flavor structure in the bar hadron.")
        hdrn_t = hdrn_state.ghdrn_t()[:-1]
    else:
        hdrn_t = hdrn_state.ghdrn_t() + 'B'
    nwqrks = []
    for i in qrks:
        if i[-1] == 'B':
            nwqrks.append(i[0])
        else:
            nwqrks.append(i + 'B')
    nwqrks.reverse()
    return hdrn(ff, hdrn_t, mntm, *nwqrks, barness = True)


# this is the main function that is going to be used to generate  the  bar
# for arbitrary state of hadrons, i.e.  single  hadron,  multihadrons  and
# also                                                      superpositions
# generate the bar      version         for         a        single-hadron
# generate the bar      version     for  a superposition of single-hadrons
# generate the bar      version for  a    h2-type operator (rank-2 tensor)
# generate   the     bar     version for    a      multi-h2-hadron   state
# generate the bar      version for  a    h3-type operator (rank-3 tensor)
# generate   the     bar     version for    a      multi-h3-hadron   state
def bar(hdrn_state):
    if isinstance(hdrn_state, hdrn):
        return barOH(hdrn_state)
    elif isinstance(hdrn_state, mhdrn):
        nwhdrns = []
        for hadron in hdrn_state.gAhdrn():
            nwhdrns.append(barOH(hadron))
        return  mhdrn(hdrn_state.glngt(), *nwhdrns)
    elif isinstance(hdrn_state, h2):
        hdrn1 = barOH(hdrn_state.ghdrn1())
        hdrn2 = barOH(hdrn_state.ghdrn2())
        return h2(hdrn2, hdrn1)
    elif isinstance(hdrn_state, mh2):
        nwh2 = []
        for tnsr in hdrn_state.gAhdrn():
            hdrn1 = barOH(tnsr.ghdrn1())
            hdrn2 = barOH(tnsr.ghdrn2())
            nwh2.append(h2(hdrn2, hdrn1))
        return mh2(hdrn_state.glngt(), *nwh2)
    elif isinstance(hdrn_state, h3):
        hdrn1 = barOH(hdrn_state.ghdrn1())
        hdrn2 = barOH(hdrn_state.ghdrn2())
        hdrn3 = barOH(hdrn_state.ghdrn3())
        nhdrnpair = hdrn3 * hdrn2
        return h3(nhdrnpair, hdrn1)
    elif isinstance(hdrn_state, mh3):
        nwh3 = []
        for tnsr in hdrn_state.gAhdrn():
            hdrn1 = barOH(tnsr.ghdrn1())
            hdrn2 = barOH(tnsr.ghdrn2())
            hdrn3 = barOH(tnsr.ghdrn3())
            nhdrnpair = hdrn3 * hdrn2
            nwh3.append(h3(nhdrnpair, hdrn1))
        return mh3(hdrn_state.glngt(), *nwh3)
    elif isinstance(hdrn_state, h4):
        hdrn1 = barOH(hdrn_state.ghdrn1())
        hdrn2 = barOH(hdrn_state.ghdrn2())
        hdrn3 = barOH(hdrn_state.ghdrn3())
        hdrn4 = barOH(hdrn_state.ghdrn4())
        nhdrntrip = hdrn4 * hdrn3 * hdrn2
        return h4(nhdrntrip, hdrn1)
    elif isinstance(hdrn_state, mh4):
        nwh4 = []
        for tnsr in hdrn_state.gAhdrn():
            hdrn1 = barOH(tnsr.ghdrn1())
            hdrn2 = barOH(tnsr.ghdrn2())
            hdrn3 = barOH(tnsr.ghdrn3())
            hdrn4 = barOH(tnsr.ghdrn4())
            nhdrntrip = hdrn4 * hdrn3 * hdrn2
            nwh4.append(h4(nhdrntrip, hdrn1))
        return mh4(hdrn_state.glngt(), *nwh4)

# this  calss  is  a  superposition     of     single-hadron     operators
# sts_n = number of states. It tells how many hadorns are in superposition
# hdrns              are                    hdrn-type              objects
# get-functions to extract all   relevant     infromations  from the mhdrn
# function      to      get      the    length    of   the   superposition
# function         to           pick       up        hadron number "nhdrn"
# function         to           give    all   hadrons in the superposition
# add        two       mhdrn       states       to       each       others
# add  one   single   hadron   to  a   superposition   of  single  hadrons
# see if  the self can be added to one of the hadrons of the superposition
# if self was not already added, then add it    to     the   superposition
# if self is already added,  then just   return  the  total  superposition
# multiply   an   mhdrn-object       state       with       a       number
# multiply one single hadron with a superposition of     single    hadrons
# divide    an    mhdrn-object       state       with       a       number
class mhdrn:
    def __init__(self, sts_n, *hdrns):
        self.sts_n = sts_n
        self.hdrns = hdrns
    def glngt(self):
        return self.sts_n
    def gOhdrn(self, nhdrn):
        return self.hdrns[nhdrn]
    def gAhdrn(self):
        return list(self.hdrns)
    def __add__(self, other):
        if isinstance(other, mhdrn):
            added_sts_n = other.glngt()  + self.glngt()
            added_hdrns = other.gAhdrn() + self.gAhdrn()
            return mhdrn(added_sts_n, *added_hdrns)
        elif isinstance(other, hdrn):
            nwmhdrn    = []
            appearence = 0
            for i, hadron in enumerate(self.gAhdrn()):
                if (hadron == other) and (appearence == 0):
                    nhdrn = hadron + other
                    nwmhdrn.append(nhdrn)
                    appearence += 1
                else:
                    nwmhdrn.append(hadron)
            if appearence == 0:

                nwlngt  = self.glngt()  + 1
                nwmhdrn.append(other)
                return mhdrn(nwlngt, *nwmhdrn)
            else:

                nwlngt  = self.glngt()
                return mhdrn(nwlngt, *nwmhdrn)
    def __radd__(self, other):
        if isinstance(other, hdrn):
            return self + other
        else:
            raise TypeError(f"Unsupported operand type(s) for +")
    def __sub__(self, other):
        if isinstance(other, mhdrn):
            return self + (-1 * other)
        elif isinstance(other, hdrn):
            return self + (-1 * other)
    def __rsub__(self, other):
        if isinstance(other, mhdrn):
            return other + (-1 * self)
        elif isinstance(other, hdrn):
            return other + (-1 * self)
        else:
            raise TypeError(f"Unsupported operand type(s) for -")
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            nwhdrns = []
            for hadron in self.gAhdrn():
                nwhdrns.append(other * hadron)
            return mhdrn(self.sts_n, *nwhdrns)
        elif isinstance(other, hdrn):
            nh2 = []
            for hadron in self.gAhdrn():
                nh2.append(hadron * other)
            return mh2(self.glngt(), *nh2)
        elif isinstance(other, mh2):
            h3container = []
            totalngth   = self.glngt() * other.glngt()
            for hadron in self.gAhdrn():
                for tnsr in other.gAhdrn():
                    h3container.append(hadron * tnsr)
            return mh3(totalngth, *h3container)
        elif isinstance(other, h2):
            h3container = []
            for hadron in self.gAhdrn():
                h3container.append(hadron * other)
            return mh3(totalngth, *h3container)
        elif isinstance(other, mhdrn):
            h2container = []
            totalngth   = self.glngt() * other.glngt()
            for hadron1 in self.gAhdrn():
                for hadron2 in other.gAhdrn():
                    h2container.append(hadron1 * hadron2)
            return mh2(totalngth, *h2container)
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, hdrn):
            nh2 = []
            for hadron in self.gAhdrn():
                nh2.append(other * hadron)
            return mh2(self.glngt(), *nh2)
        elif isinstance(other, mh2):
            h3container = []
            totalngth   = self.glngt() * other.glngt()
            for hadron in self.gAhdrn():
                for tnsr in other.gAhdrn():
                    h3container.append(tnsr * hadron)
            return mh3(totalngth, *h3container)
        elif isinstance(other, h2):
            h3container = []
            for hadron in self.gAhdrn():
                h3container.append(other * hadron)
            return mh3(totalngth, *h3container)
        elif isinstance(other, mhdrn):
            h2container = []
            totalngth   = self.glngt() * other.glngt()
            for hadron1 in self.gAhdrn():
                for hadron2 in other.gAhdrn():
                    h2container.append(hadron2 * hadron1)
            return mh2(totalngth, *h2container)
        else:
            raise TypeError(f"wrong multiplication with multihadron state")
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            nwhdrns = []
            for hadron in self.gAhdrn():
                nwhdrns.append(hadron/other)
            return mhdrn(self.sts_n, *nwhdrns)
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            nwhdrns = []
            for hadron in self.gAhdrn():
                nwhdrns.append(hadron/other)
            return mhdrn(self.sts_n, *nwhdrns)
        else:
            raise TypeError(f"wrong division with multihadron state")



# a class  to  generate     a     two-hadron     state     (rank-2 tensor)
# hdrn1 is the first  hadron  while      hdrn2 is the      second      one
# ff1  and   ff2   are   the   numerical   factors  inside hdrn1 and hdrn2
# functions to  get the   individual   hadrons   of   the   tensor product
# add one zwhdrn-type object to a  superposition  of  zwhdrn-type  objects
# multiply     a       single       h2-type      with        a      number
# multiply one single hadron with  rank-2    tensor (i.e.m a h2-object) to 
# build  a     rank-3     tensor    i.e.,    a   three-hadron state object
# of                 the                    type                        h3
# Divide       a          single       h2-type       with      a    number
class h2:
    def __init__(self, hdrn1, hdrn2):
        self.hdrn1   = hdrn1
        self.hdrn2   = hdrn2
        self.ff1     = (self.hdrn1).gff()
        self.ff2     = (self.hdrn2).gff()
    def gsprnf(self):
        return self.ff1 * self.ff2
    def ghdrn1(self):
        return hdrn(self.ff1 * self.ff2, (self.hdrn1).ghdrn_t(), (self.hdrn1).gmntm(), *(self.hdrn1).gqrks(), barness = (self.hdrn1).gbarness())
    def ghdrn2(self):
        return hdrn(1, (self.hdrn2).ghdrn_t(), (self.hdrn2).gmntm(), *(self.hdrn2).gqrks(), barness = (self.hdrn2).gbarness())
    def gAhdrn(self):
        return [self.ghdrn1(), self.ghdrn2()]
    def __eq__(self, other):
        return isinstance(other, h2) and (self.ghdrn1() == other.ghdrn1()) and (self.ghdrn2() == other.ghdrn2())
    def __add__(self, other):
        if isinstance(other, h2):
            if self == other:
                nhdrn1 = self.ghdrn1() + other.ghdrn1()
                return h2(nhdrn1, self.ghdrn2())
            else:
                return mh2(2, self, other)
        elif isinstance(other, mh2):
            nwh2    = []
            appearence = 0
            for i, towhadrons in enumerate(other.gAhdrn()):

                if (towhadrons == self) and (appearence == 0):
                    nthdrns = towhadrons + self
                    nwh2.append(nthdrns)
                    appearence += 1
                else:
                    nwh2.append(towhadrons)
            if appearence == 0:
                nwlngt  = other.glngt()  + 1
                nwh2.append(self)
                return mh2(nwlngt, *nwh2)
            else:
                nwlngt  = other.glngt()
                return mh2(nwlngt, *nwh2)
    def __radd__(self, other):
        if isinstance(other, mh2):
            return self + other
        else:
            raise TypeError(f"Unsupported operand type(s) for +")
    def __sub__(self, other):
        if isinstance(other, h2):
            return self + (-1 * other)
        elif isinstance(other, mh2):
            return self + (-1 * other)
    def __rsub__(self, other):
        if isinstance(other, h2):
            return other + (-1 * self)
        elif isinstance(other, mh2):
            return other + (-1 * self)
        else:
            raise TypeError(f"Unsupported operand type(s) for -")
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            nhdrn1 = other * self.ghdrn1()
            return h2(nhdrn1, self.ghdrn2())
        elif isinstance(other, hdrn):
            return h3(self, other)
        elif isinstance(other, mhdrn):
            h3container = []
            totalngth = other.glngt()
            for hadron in other.gAhdrn():
                h3container.append(self * hadron)
            return mh3(totalngth, *h3container)
        elif isinstance(other, h2):
            hdrn_trip = self.ghdrn1() * self.ghdrn2() * other.ghdrn1()
            return h4(hdrn_trip, other.ghdrn2())
        elif isinstance(other, mh2):
            h4container = []
            totalngth = other.glngt()
            for two_hadron in other.gAhdrn():
                h4container.append(self * two_hadron)
            return mh4(totalngth, *h4container)
        elif isinstance(other, mhdrn):
            h4container = []
            totalngth   = self.glngt() * other.glngt()
            for tnsr1 in self.gAhdrn():
                for tnsr2 in other.gAhdrn():
                    h4container.append(tnsr1 * tnsr2)
            return mh4(totalngth, *h4container)
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, hdrn):
            hdrnpair = h2(other, self.ghdrn1())
            return h3(hdrnpair, self.ghdrn2())
        elif isinstance(other, mhdrn):
            h3container = []
            totalngth = other.glngt()
            for hadron in other.gAhdrn():
                h3container.append(hadron * self)
            return mh3(totalngth, *h3container)
        elif isinstance(other, h2):
            hdrn_trip = other.ghdrn1() * other.ghdrn2() * self.ghdrn1()
            return h4(hdrn_trip, self.ghdrn2())
        elif isinstance(other, mh2):
            h4container = []
            totalngth = other.glngt()
            for two_hadron in other.gAhdrn():
                h4container.append(two_hadron * self)
            return mh4(totalngth, *h4container)
        elif isinstance(other, mhdrn):
            h4container = []
            totalngth   = self.glngt() * other.glngt()
            for tnsr1 in self.gAhdrn():
                for tnsr2 in other.gAhdrn():
                    h4container.append(tnsr2 * tnsr1)
            return mh4(totalngth, *h4container)
        raise TypeError(f"Unsupported operand type(s) for *")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            nhdrn1 = self.ghdrn1() / other
            return h2(nhdrn1, self.ghdrn2())
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self / other
        else:
            raise TypeError(f"Unsupported operand type(s) for /")




# this  class  is  a  superposition      of      two-hadrons     operators
# it is built similar to    class of superposition of single hadrons mhdrn
class mh2:
    def __init__(self, sts_n, *h2s):
        self.sts_n = sts_n
        self.h2s = h2s
    def glngt(self):
        return self.sts_n
    def gOhdrn(self, nhdrn):
        return self.h2s[nhdrn]
    def gAhdrn(self):
        return list(self.h2s)
    def __add__(self, other):
        if isinstance(other, mh2):
            added_sts_n = other.glngt()  + self.glngt()
            added_hdrns = other.gAhdrn() + self.gAhdrn()
            return mh2(added_sts_n, *added_hdrns)
        elif isinstance(other, h2):
            nwh2    = []
            appearence = 0
            for i, tnsor in enumerate(self.gAhdrn()):
                if (tnsor == other) and (appearence == 0):
                    nhdrn = tnsor + other
                    nwh2.append(nhdrn)
                    appearence += 1
                else:
                    nwh2.append(tnsor)
            if appearence == 0:
                nwlngt  = self.glngt()  + 1
                nwh2.append(other)
                return mh2(nwlngt, *nwh2)
            else:
                nwlngt  = self.glngt()
                return mh2(nwlngt, *nwh2)
    def __radd__(self, other):
        if isinstance(other, h2):
            return self + other
        else:
            raise TypeError(f"Unsupported operand type(s) for +")
    def __sub__(self, other):
        if isinstance(other, mh2):
            return self + (-1 * other)
        elif isinstance(other, h2):
            return self + (-1 * other)
    def __rsub__(self, other):
        if isinstance(other, mh2):
            return other + (-1 * self)
        elif isinstance(other, h2):
            return other + (-1 * self)
        else:
            raise TypeError(f"Unsupported operand type(s) for -")
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            nwh2s = []
            for tnsor in self.gAhdrn():
                nhdrn1 = tnsor.ghdrn1() * other
                nwh2s.append(h2(nhdrn1, tnsor.ghdrn2()))
            return mh2(self.sts_n, *nwh2s)
        elif isinstance(other, hdrn):
            nh3 = []
            for two_hadrons in self.gAhdrn():
                nh3.append(two_hadrons * other)
            return mh3(self.glngt(), *nh3)
        elif isinstance(other, mh1):
            h3container = []
            totalngth   = self.glngt() * other.glngt()
            for hadron in other.gAhdrn():
                for tnsr in self.gAhdrn():
                    h3container.append(tnsr * hadron)
            return mh3(totalngth, *h3container)
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, hdrn):
            nh3 = []
            for two_hadrons in self.gAhdrn():
                nh3.append(other * two_hadrons)
            return mh3(self.glngt(), *nh3)
        elif isinstance(other, mh1):
            h3container = []
            totalngth   = self.glngt() * other.glngt()
            for hadron in other.gAhdrn():
                for tnsr in self.gAhdrn():
                    h3container.append(hadron * tnsr)
            return mh3(totalngth, *h3container)
        else:
            raise TypeError(f"wrong multiplication with multihadron state")
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            nwh2s = []
            for tnsor in self.gAhdrn():
                nhdrn1 = tnsor.ghdrn1() / other
                nwh2s.append(h2(nhdrn1, tnsor.ghdrn2()))
            return mh2(self.sts_n, *nwh2s)
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self / other
        else:
            raise TypeError(f"wrong division with multihadron state")


# a class  to  generate     a    three-hadron     state     (rank-3 tensor)
# built similar     to     a        two-hadron       single      state   h2
class h3:
    def __init__(self, hdrn_pair, hdrn3):
        self.hdrn_pair  = hdrn_pair
        self.hdrn1      = (self.hdrn_pair).ghdrn1()
        self.hdrn2      = (self.hdrn_pair).ghdrn2()
        self.hdrn3      = hdrn3
        self.ff1        = (self.hdrn1).gff()
        self.ff2        = (self.hdrn2).gff()
        self.ff3        = (self.hdrn3).gff()
        self.totalff    = self.ff1 * self.ff2 * self.ff3
    def gsprnf(self):
        return self.totalff
    def ghdrn1(self):
        return hdrn(self.totalff, (self.hdrn1).ghdrn_t(), (self.hdrn1).gmntm(), *(self.hdrn1).gqrks(), barness = (self.hdrn1).gbarness())
    def ghdrn2(self):
        return hdrn(1, (self.hdrn2).ghdrn_t(), (self.hdrn2).gmntm(), *(self.hdrn2).gqrks(), barness = (self.hdrn2).gbarness())
    def ghdrn3(self):
        return hdrn(1, (self.hdrn3).ghdrn_t(), (self.hdrn3).gmntm(), *(self.hdrn3).gqrks(), barness = (self.hdrn3).gbarness())
    def gAhdrn(self):
        return [self.ghdrn1(), self.ghdrn2(), self.ghdrn3()]
    def __eq__(self, other):
        return isinstance(other, h3) and self.ghdrn1() == other.ghdrn1() and self.ghdrn2() == other.ghdrn2() and self.ghdrn3() == other.ghdrn3()
    def __add__(self, other):
        if isinstance(other, h3):
            if self == other:
                nhdrn1 = self.ghdrn1() + other.ghdrn1()
                nhdrn_pair = nhdrn1 * self.ghdrn2()
                return h3(nhdrn_pair, self.ghdrn3())
            else:
                return mh3(2, self, other)
        elif isinstance(other, mh3):
            nwh3    = []
            appearence = 0
            for i, thrhadrons in enumerate(other.gAhdrn()):
                if (thrhadrons == self) and (appearence == 0):
                    nthdrns = thrhadrons + self
                    nwh3.append(nthdrns)
                    appearence += 1
                else:
                    nwh3.append(thrhadrons)
            if appearence == 0:
                nwlngt  = other.glngt()  + 1
                nwh3.append(self)
                return mh3(nwlngt, *nwh3)
            else:
                nwlngt  = other.glngt()
                return mh3(nwlngt, *nwh3)
    def __radd__(self, other):
        if isinstance(other, mh3):
            return self + other
        else:
            raise TypeError(f"Unsupported operand type(s) for +")
    def __sub__(self, other):
        if isinstance(other, h3):
            return self + (-1 * other)
        elif isinstance(other, mh3):
            return self + (-1 * other)

    def __rsub__(self, other):
        if isinstance(other, h3):
            return other + (-1 * self)
        elif isinstance(other, mh3):
            return other + (-1 * self)
        else:
            raise TypeError(f"Unsupported operand type(s) for -")
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            nhdrn1     = other * self.ghdrn1()
            nhdrn_pair = nhdrn1 * self.ghdrn2()
            return h3(nhdrn_pair, self.ghdrn3())
        elif isinstance(other, hdrn):
            return h4(self, other)
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, hdrn):
            return h4(other * self.ghdrn1() * self.ghdrn2(), self.ghdrn3())
        else:
            raise TypeError(f"Unsupported operand type(s) for *")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            nhdrn1     =  self.ghdrn1() / other
            nhdrn_pair = nhdrn1 * self.ghdrn2()
            return h3(nhdrn_pair, self.ghdrn3())
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self / other
        else:
            raise TypeError(f"Unsupported operand type(s) for /")






# this  class  is  a  superposition      of    three-hadrons     operators
# it is built similar to    class of superposition of single hadrons mhdrn
class mh3:
    def __init__(self, sts_n, *h3s):
        self.sts_n = sts_n
        self.h3s = h3s
    def glngt(self):
        return self.sts_n
    def gOhdrn(self, nhdrn):
        return self.h3s[nhdrn]
    def gAhdrn(self):
        return list(self.h3s)
    def __add__(self, other):
        if isinstance(other, mh3):
            added_sts_n = other.glngt()  + self.glngt()
            added_hdrns = other.gAhdrn() + self.gAhdrn()
            return mh3(added_sts_n, *added_hdrns)
        elif isinstance(other, h3):
            nwh3    = []
            appearence = 0
            for i, thrhadrons in enumerate(self.gAhdrn()):
                if (thrhadrons == other) and (appearence == 0):
                    nthdrns = thrhadrons + other
                    nwh3.append(nthdrns)
                    appearence += 1
                else:
                    nwh3.append(thrhadrons)
            if appearence == 0:
                nwlngt  = self.glngt()  + 1
                nwh3.append(other)
                return mh3(nwlngt, *nwh3)
            else:
                nwlngt  = self.glngt()
                return mh3(nwlngt, *nwh3)
    def __radd__(self, other):
        if isinstance(other, h3):
            return self + other
        else:
            raise TypeError(f"Unsupported operand type(s) for +")
    def __sub__(self, other):
        if isinstance(other, mh3):
            return self + (-1 * other)
        elif isinstance(other, h3):
            return self + (-1 * other)
    def __rsub__(self, other):
        if isinstance(other, mh3):
            return other + (-1 * self)
        elif isinstance(other, h3):
            return other + (-1 * self)
        else:
            raise TypeError(f"Unsupported operand type(s) for -")
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            nwh3s = []
            for tnsor in self.gAhdrn():
                nhdrn1     = other * tnsor.ghdrn1()
                nhdrn_pair = nhdrn1 * tnsor.ghdrn2()
                nwh3s.append(h3(nhdrn_pair, tnsor.ghdrn3()))
            return mh3(self.sts_n, *nwh3s)
        elif isinstance(other, hdrn):
            nh4 = []
            for triplet_hadrons in self.gAhdrn():
                nh4.append(triplet_hadrons * other)
            return mh4(self.glngt(), *nh4)
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, hdrn):
            nh4 = []
            for triplet_hadrons in self.gAhdrn():
                nh4.append(other * triplet_hadrons)
            return mh4(self.glngt(), *nh4)
        else:
            raise TypeError(f"wrong multiplication with multihadron state")
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            nwh3s = []
            for tnsor in self.gAhdrn():
                nhdrn1     = tnsor.ghdrn1() / other
                nhdrn_pair = nhdrn1 * tnsor.ghdrn2()
                nwh3s.append(h3(nhdrn_pair, tnsor.ghdrn3()))
            return mh3(self.sts_n, *nwh3s)
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self / other
        else:
            raise TypeError(f"wrong division with multihadron state")












class h4:
    def __init__(self, hdrn_trip, hdrn4):
        self.hdrn_trip  = hdrn_trip
        self.hdrn1      = (self.hdrn_trip).ghdrn1()
        self.hdrn2      = (self.hdrn_trip).ghdrn2()
        self.hdrn3      = (self.hdrn_trip).ghdrn3()
        self.hdrn4      = hdrn4
        self.ff1        = (self.hdrn1).gff()
        self.ff2        = (self.hdrn2).gff()
        self.ff3        = (self.hdrn3).gff()
        self.ff4        = (self.hdrn4).gff()
        self.totalff    = self.ff1 * self.ff2 * self.ff3 * self.ff4
    def gsprnf(self):
        return self.totalff
    def ghdrn1(self):
        return hdrn(self.totalff, (self.hdrn1).ghdrn_t(), (self.hdrn1).gmntm(), *(self.hdrn1).gqrks(), barness = (self.hdrn1).gbarness())
    def ghdrn2(self):
        return hdrn(1, (self.hdrn2).ghdrn_t(), (self.hdrn2).gmntm(), *(self.hdrn2).gqrks(), barness = (self.hdrn2).gbarness())
    def ghdrn3(self):
        return hdrn(1, (self.hdrn3).ghdrn_t(), (self.hdrn3).gmntm(), *(self.hdrn3).gqrks(), barness = (self.hdrn3).gbarness())
    def ghdrn4(self):
        return hdrn(1, (self.hdrn4).ghdrn_t(), (self.hdrn4).gmntm(), *(self.hdrn4).gqrks(), barness = (self.hdrn4).gbarness())
    def gAhdrn(self):
        return [self.ghdrn1(), self.ghdrn2(), self.ghdrn3(), self.ghdrn4()]
    def __eq__(self, other):
        return isinstance(other, h3) and self.ghdrn1() == other.ghdrn1() and self.ghdrn2() == other.ghdrn2() and self.ghdrn3() == other.ghdrn3() and self.ghdrn4() == other.ghdrn4()
    def __add__(self, other):
        if isinstance(other, h4):
            if self == other:
                nhdrn1 = self.ghdrn1() + other.ghdrn1()
                nhdrn_trip = nhdrn1 * self.ghdrn2() * self.ghdrn3() 
                return h3(nhdrn_trip, self.ghdrn4())
            else:
                return mh4(2, self, other)
        elif isinstance(other, mh4):
            nwh4    = []
            appearence = 0
            for i, thrhadrons in enumerate(other.gAhdrn()):
                if (thrhadrons == self) and (appearence == 0):
                    nthdrns = thrhadrons + self
                    nwh4.append(nthdrns)
                    appearence += 1
                else:
                    nwh4.append(thrhadrons)
            if appearence == 0:
                nwlngt  = other.glngt()  + 1
                nwh4.append(self)
                return mh4(nwlngt, *nwh4)
            else:
                nwlngt  = other.glngt()
                return mh4(nwlngt, *nwh4)
    def __radd__(self, other):
        if isinstance(other, mh4):
            return self + other
        else:
            raise TypeError(f"Unsupported operand type(s) for +")
    def __sub__(self, other):
        if isinstance(other, h4):
            return self + (-1 * other)
        elif isinstance(other, mh4):
            return self + (-1 * other)
    def __rsub__(self, other):
        if isinstance(other, h4):
            return other + (-1 * self)
        elif isinstance(other, mh4):
            return other + (-1 * self)
        else:
            raise TypeError(f"Unsupported operand type(s) for -")
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            nhdrn1     = other * self.ghdrn1()
            nhdrn_trip = nhdrn1 * self.ghdrn2() * self.ghdrn3()
            return h4(nhdrn_trip, self.ghdrn4())
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        else:
            raise TypeError(f"Unsupported operand type(s) for *")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            nhdrn1     =  self.ghdrn1() / other
            nhdrn_trip = nhdrn1 * self.ghdrn2() * self.ghdrn3()
            return h4(nhdrn_trip, self.ghdrn4())
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self / other
        else:
            raise TypeError(f"Unsupported operand type(s) for /")




class mh4:
    def __init__(self, sts_n, *h4s):
        self.sts_n = sts_n
        self.h4s = h4s
    def glngt(self):
        return self.sts_n
    def gOhdrn(self, nhdrn):
        return self.h4s[nhdrn]
    def gAhdrn(self):
        return list(self.h4s)
    def __add__(self, other):
        if isinstance(other, mh4):
            added_sts_n = other.glngt()  + self.glngt()
            added_hdrns = other.gAhdrn() + self.gAhdrn()
            return mh4(added_sts_n, *added_hdrns)
        elif isinstance(other, h4):
            nwh4    = []
            appearence = 0
            for i, thrhadrons in enumerate(self.gAhdrn()):
                if (thrhadrons == other) and (appearence == 0):
                    nthdrns = thrhadrons + other
                    nwh4.append(nthdrns)
                    appearence += 1
                else:
                    nwh4.append(thrhadrons)
            if appearence == 0:
                nwlngt  = self.glngt()  + 1
                nwh4.append(other)
                return mh4(nwlngt, *nwh4)
            else:
                nwlngt  = self.glngt()
                return mh4(nwlngt, *nwh4)
    def __radd__(self, other):
        if isinstance(other, h4):
            return self + other
        else:
            raise TypeError(f"Unsupported operand type(s) for +")
    def __sub__(self, other):
        if isinstance(other, mh4):
            return self + (-1 * other)
        elif isinstance(other, h4):
            return self + (-1 * other)
    def __rsub__(self, other):
        if isinstance(other, mh4):
            return other + (-1 * self)
        elif isinstance(other, h4):
            return other + (-1 * self)
        else:
            raise TypeError(f"Unsupported operand type(s) for -")
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            nwh4s = []
            for tnsor in self.gAhdrn():
                nhdrn1     = other * tnsor.ghdrn1()
                nhdrn_trip = nhdrn1 * tnsor.ghdrn2() * tnsor.ghdrn3()
                nwh4s.append(h4(nhdrn_trip, tnsor.ghdrn4()))
            return mh4(self.sts_n, *nwh4s)
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        else:
            raise TypeError(f"wrong multiplication with multihadron state")
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            nwh4s = []
            for tnsor in self.gAhdrn():
                nhdrn1     = tnsor.ghdrn1() / other
                nhdrn_trip = nhdrn1 * tnsor.ghdrn2() * tnsor.ghdrn3()
                nwh4s.append(h4(nhdrn_trip, tnsor.ghdrn4()))
            return mh4(self.sts_n, *nwh4s)
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self / other
        else:
            raise TypeError(f"wrong division with multihadron state")





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
    state = {1: hdrn(1, 'xi', mntm, 's', 's', 'u', barness = fls), -1:  hdrn(-1, 'xi', mntm, 's', 's', 'd', barness = fls)}
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
    state = {'+': hdrn(1, 'K', mntm, 'sB', 'u', barness = fls), 0:  hdrn(1, 'K', mntm, 'sB', 'd', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of kaon must be the its type, i.e. +, - or 0 ")

def KaonC(ispin, mntm1 = None):
    if mntm1 == None:
        mntm = 1
    else:
        mntm = mntm1
    state = {'+': hdrn(1, 'Kc', mntm, 'dB', 's', barness = fls), 0:  hdrn(-1, 'Kc', mntm, 'uB', 's', barness = fls)}
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
            , 0:  hdrn(1/2**(1/2), 'Pi0', mntm, 'dB', 'd', barness = fls) + hdrn(-1/2**(1/2), 'Pi0', mntm, 'uB', 'u', barness = fls)}
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

def Phi(ispin, mntm1 = None):
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
    state = {'+': hdrn(1, 'D+', mntm, 'dB', 'c', barness = fls), '-':  hdrn(1, 'D-', mntm, 'cB', 'd', barness = fls)
            , 0:  hdrn(1, 'D0', mntm, 'uB', 'c', barness = fls)}
    if ispin in state:
        return state[ispin]
    else:
        raise TypeError(f"Error: First argument of DMeson must be the its type, i.e. +, - or 0 ")



################################################################################################
######    now     we   define operators which  live in some specific   time     slices    ######
################################################################################################
#0 ist source, 1 ist sink

#Go_Through_All_Again and perform Simplification for a diagram_container!
#Make add more consistent
#Generate plots
class dgrm_container:
    def __init__(self, *dgrms):
        self.dgrms = dgrms
    def gdiagrams (self): 
        return [diagram for diagram in list(self.dgrms) if isinstance(diagram, dgrm) and diagram.gff() != 0]
    def gdgrms(self):
        sp1      = list(self.dgrms)
        simsp1   = []
        seen     = set()
        counter  = 0
        print(f"Start simplifying {len(sp1)} diagrams")
        for i1 in range(len(sp1)):
            if i1 in seen:
                continue
            else:
                simsp1.append(sp1[i1])
            for i2 in range(i1 + 1, len(sp1)):
                if i2 in seen:
                    continue
                if sp1[i1] == sp1[i2]:
                    counter += 1
                    simsp1[-1] = simsp1[-1] + sp1[i2]
                    seen.add(i2)
            if i1 % 100 == 0 and i1 != 0:
                print(f"*** Simplifying went through diagram {i1} ***")
        print(f" {counter} diagrams have been reduced")
        return [diagram for diagram in simsp1 if isinstance(diagram, dgrm) and diagram.gff() != 0]
    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        elif isinstance(other, dgrm_container):
            ndgrms = self.gdiagrams() + other.gdiagrams()
            return dgrm_container(*ndgrms)
        elif isinstance(other, dgrm):
            ndgrms  = self.gdiagrams() + [other]
            return dgrm_container(*ndgrms)
    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self + other
        elif isinstance(other, dgrm):
            return self + other
        else:
            raise TypeError("Undefined addition +operator")

class prpgtr:
# a class to store the contracted quark    pair    into    a    propagator
    def __init__(self, factor, quark1, quark2):
        self.quark1 = quark1
        self.quark2 = quark2
        self.factor = factor
    def gqrk1(self):
        return self.quark1
    def gqrk2(self):
        return self.quark2
    def ghdrn1(self):
        return self.quark1.ghdrn_ontime()
    def ghdrn2(self):
        return self.quark2.ghdrn_ontime()
    def gbar(self):
        if (self.quark1).gflvr()[-1] == 'B':
            return self.quark1
        else:
            return self.quark2
    def gnbar(self):
        if (self.quark1).gflvr()[-1] != 'B':
            return self.quark1
        else:
            return self.quark2
    def gHbar(self):
        return self.gbar().ghdrn_ontime()
    def gHnbar(self):
        return self.gnbar().ghdrn_ontime()
    def gff(self):
        return self.factor
    def numbering(self):
        return [self.gnbar().gnmbr(), self.gbar().gnmbr()]
    def __mul__(self, other):
        if isinstance(other, prpgtr):
            nff = self.gff() * other.gff()
            return dgrm(nff, *(self, other))
        elif isinstance(other, dgrm):
            nwprpgtrs = other.gpropagators() + [self]
            nff = self.gff()
            for propagator in nwprpgtrs:
                nff *= propagator.gff()
            return dgrm(nff, *nwprpgtrs)
    def __rmul__(self, other):
        if isinstance(other, dgrm):
            return self * other
        else:
            raise TypeError(f"Error: Undefined multiplication with a propagator")
    def gproperties(self):
        return self.gnbar().gqrktprp() + self.gbar().gqrktprp()
    def __eq__(self, other):
        return isinstance(other, prpgtr) and self.gproperties() == other.gproperties()

def sgn_dgrm(*prpgtrs):
    permuted_elements = []
    for propagator in prpgtrs:
        if not isinstance(propagator, prpgtr):
            raise TypeError("You are trying to calculate the grassmann sing for a non-propagator-typed object")
        order    = propagator.numbering()
        quark    = order[0]
        quarkbar = order[1]
        permuted_elements.append(quark)
        permuted_elements.append(quarkbar)
    inversions = 0
    nF = len(permuted_elements)
    for i in range(nF):
        for j in range(i + 1, nF):
            if permuted_elements[i] > permuted_elements[j]:
                inversions += 1
    sign = (-1) ** inversions
    return sign
def tl_ff_prp(*prpgtrs):
    nff = 1
    for propagator in prpgtrs:
        nff *= propagator.gff()
    return nff

def einzigartige_prpgtrs(*prpgtrs):
    for propagator in prpgtrs:
        if not isinstance(propagator, prpgtr):
            raise TypeError("Problem with comparing propagators")
    gesehene_quarks = []

    for prp in prpgtrs:
        q1, q2 = prp.gqrk1(), prp.gqrk2()
        if q1 in gesehene_quarks or q2 in gesehene_quarks:
            return False
        gesehene_quarks.append(q1)
        gesehene_quarks.append(q2)
    return True

def deep_tuple(lst):
    return tuple(deep_tuple(item) if isinstance(item, list) else item for item in lst)

class dgrm:
    def __init__(self, factor, *prpgtrs):
        self.prpgtrs = prpgtrs
        self.factor  = factor
    def gpropagators(self):
        return list(self.prpgtrs)
    def gff(self):
        return self.factor
    # mehr funktionen um Informationen rauszuholen
    def __mul__(self, other):
        if isinstance(other, dgrm):
            nwprpgtrs = self.gpropagators() + other.gpropagators()
            nff = 1
            for propagator in nwprpgtrs:
                nff *= propagator.gff()
            return dgrm(nff, *nwprpgtrs)
        elif isinstance(other, prpgtr):
            nwprpgtrs = self.gpropagators() + [other]
            nff = 1
            for propagator in nwprpgtrs:
                nff *= propagator.gff()
            return dgrm(nff, *nwprpgtrs)
        elif isinstance(other, int):
            nff = other * self.gff()
            return dgrm(nff, *self.gpropagators())
    def __rmul__(self, other):
        if isinstance(other, prpgtr):
            return self * other
        elif isinstance(other, int):
            return self * other
        else:
            raise TypeError(f"Error: Undefined multiplication with a diagram")
    def bulavision(self):
        list_of_propagators = []
        for propagator in self.gpropagators():
            q1 = propagator.gbar()
            q2 = propagator.gnbar()
            list_of_propagators.append([[q2.gtm(), q2.ghdrn_n(), q2.gqrk_hdrn_p()], [q1.gtm(), q1.ghdrn_n(), q1.gqrk_hdrn_p()]])
        print("numerical factor = ", self.gff())
        print("Topology: ")
        print(list_of_propagators)
        diagramplot(self)
        return list_of_propagators
        print("************************************************")
    def __add__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        elif isinstance(other, dgrm):
            if self == other:
                nff = self.gff() + other.gff()
                if nff == 0:
                    return 0
                else:
                    return dgrm(nff, *self.gpropagators())
            else:
                return dgrm_container(self, other)
        elif isinstance(other, dgrm_container):
            ndgrms  = other.gdiagrams() + [self]
            return dgrm_container(*ndgrms)
            #else:
            #    raise TypeError("Error: Failed to add a diagram to a set of diagrams")
    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        elif isinstance(other, dgrm):
            return self + other
        elif isinstance(other, dgrm_container):
            return self + other
        else:
            raise TypeError("Undefined addition +operator with diagrams")
    def __eq__(self, other):
        if isinstance(other, dgrm) and len(self.gpropagators()) == len(other.gpropagators()):
            prps_in_self  = tuple(propagator.gproperties() for propagator in self.gpropagators())
            prps_in_other = tuple(propagator.gproperties() for propagator in other.gpropagators())
            #prps_in_self  = tuple(item[0] if isinstance(item, tuple) and len(item) == 1 else item for item in prps_in_self)
            #prps_in_other = tuple(item[0] if isinstance(item, tuple) and len(item) == 1 else item for item in prps_in_other)
            return set(prps_in_self) == set(prps_in_other)
        else:
            return False



# creatogram         takes           quark     line   of      the     form
# {flvr1:[qrks...], flvr2:[qrks...], ....} and creates out   of  it    all
# possible diagrams and gives them at the end in form of a  dgrm_container
# object. it proceeds                      as                      follows
# it uses contractor to 1. generate a list of all possible propagators for
# each                                                              flavor
# one list of list1, list2 contains some specific quark type and the other
# list contains the the same quark type but in bar, both  lists  have  the
# same                                                              length
# now in contracted_list we have all possible   propagators, this now must
# be filtered to propagators that appear in the same term and those  which
# do not    appear     in         the          same        term     (line)
# by that we mean: if we have (q1 * q2_B) * (q3 * q4_B), then so far   the
# output of contractor would be [P12, P14, P32], however in the    diagram
# we construct later, P12 and  P14 cannot appear in the    same   diagram!
# so to construct diagrams out of contracted_list, we see how many  quarks
# we have, then the the diagram has the length of the quarks/2, so we take
# all combinations of propagators, for which non of the propagators    are
# equal                  to                  each                   others
# the output of contractor is now a list, whose elements are superpositons
# a map containing all sets of propagators according  to    the     flavor
# at the end sets_of_flvr[flavor] contains a map of propagators multiplied
# each line in this map contains set of propagators in superpositions,i.e.
# the final result of the diagram is obtained by multiplying all     lines
# of   different      flavors           with           each         others
# here we have only one flavor, i.e. the final result is obtained       by
# summing the diagrams in the list   contractor     to     each     others
# diagram is a dgrm_container-typed oject, i.e. it represents the  sum  of
# all diagrams and hence the final result before doing  the simplification
def creatograms(quarkline):
    def contractor(list1, list2):
        if len(list1) != len(list2):
            raise ErrorType("unequal number of q and qbar")
        number_of_prp = len(list1)
        contracted_list = []
        for quark1 in list1:
            for quark2 in list2:
                nff = quark1.gff() * quark2.gff()
                contracted_list.append(prpgtr(nff, quark1, quark2))
        return [dgrm(tl_ff_prp(*komb), *komb) for komb in combinations(contracted_list, number_of_prp) if einzigartige_prpgtrs(*komb)]
    sets_of_flvr = {}
    flavorlist   = []
    for flavor in quarkline:
        if flavor[-1] != 'B':
            flavorlist.append(flavor)
    print(f"We have following flavors {flavorlist}")
    for flavor in flavorlist:
        flavorB = flavor + 'B'
        print(f"Obtain propagators for the {flavor} quarks...")
        sets_of_flvr[flavor] = contractor(quarkline[flavor], quarkline[flavorB])
        if len(sets_of_flvr[flavor]) == 0:
            raise TypeError("Failed to remove redundant correlators")
        print("Done.")
    final_diagram_list = []
    print("Multiply now the propagators with each others (redundant ones are not considered) and ")
    print("assign with each diagram the correct sign due to Grassmann-Algebra")
    if len(flavorlist) == 1:
        flvr1 = flavorlist[0]
        for diagram in sets_of_flvr[flvr1]:
            final_diagram_list.append(diagram)
        for i in range(len(final_diagram_list)):
            dgrmsing              = sgn_dgrm(*final_diagram_list[i].gpropagators())
            nff                   = dgrmsing * final_diagram_list[i].gff()
            final_diagram_list[i] = dgrm(nff, *final_diagram_list[i].gpropagators())
        print("Done.")
    elif len(flavorlist) == 2:
        flvr1 = flavorlist[0]
        flvr2 = flavorlist[1]
        for diagram1 in sets_of_flvr[flvr1]:
            for diagram2 in sets_of_flvr[flvr2]:
                final_diagram_list.append(diagram1 * diagram2)
        for i in range(len(final_diagram_list)):
            dgrmsing              = sgn_dgrm(*final_diagram_list[i].gpropagators())
            nff                   = dgrmsing * final_diagram_list[i].gff()
            final_diagram_list[i] = dgrm(nff, *final_diagram_list[i].gpropagators())
        print("Done.")
    elif len(flavorlist) == 3:
        flvr1 = flavorlist[0]
        flvr2 = flavorlist[1]
        flvr3 = flavorlist[2]
        final_diagram_list = []
        for diagram1 in sets_of_flvr[flvr1]:
            for diagram2 in sets_of_flvr[flvr2]:
                for diagram3 in sets_of_flvr[flvr3]:
                    final_diagram_list.append(diagram1 * diagram2 * diagram3)
        for i in range(len(final_diagram_list)):
            dgrmsing              = sgn_dgrm(*final_diagram_list[i].gpropagators())
            nff                   = dgrmsing * final_diagram_list[i].gff()
            final_diagram_list[i] = dgrm(nff, *final_diagram_list[i].gpropagators())
        print("Done.")
    elif len(flavorlist) == 4:
        flvr1 = flavorlist[0]
        flvr2 = flavorlist[1]
        flvr3 = flavorlist[2]
        flvr4 = flavorlist[3]
        final_diagram_list = []
        for diagram1 in sets_of_flvr[flvr1]:
            for diagram2 in sets_of_flvr[flvr2]:
                for diagram3 in sets_of_flvr[flvr3]:
                    for diagram4 in sets_of_flvr[flvr4]:
                        final_diagram_list.append(diagram1 * diagram2 * diagram3 * diagram4)
        for i in range(len(final_diagram_list)):
            dgrmsing              = sgn_dgrm(*final_diagram_list[i].gpropagators())
            nff                   = dgrmsing * final_diagram_list[i].gff()
            final_diagram_list[i] = dgrm(nff, *final_diagram_list[i].gpropagators())
        print("Done.")
    else:
        print("Ask Herz to update the wicktract-code")
        print("to involve more than 4 different flavors at the same time!")
        raise TypeError("Error!")
    print("Done.")
    print(f"In total, for this correlator there are {len(final_diagram_list)} diagrams")
    print(" ")
    return dgrm_container(*final_diagram_list)






# a class which takes states of hadronsors and associate each hadron  with
# the corresponding number and time. In case the     hadrons    are     in
# superposition, it generates a list of each hadrons in each term  of  the
# superposition. This is important step before performing the contractions
# superposition, it generates a list of each hadrons in each term  of  the
# tm corresponds to the time, 0 for source, 1 for        sink and anything
# for                         the                                current/s
# hdrn_state  is   an  object  of  the  type hdrn/mhdrn/h2/mh2/h3  or  mh3
# stntme means a  hadronic  state on   a given   time, i.e.  state_on_time
# stntme will be of the form {0: [hdrn, hdrn,..], 1:[hdrn, hdrn, ..], ...}
# the numbers correspond to  the  numbering  in  the  superposition,  e.g.
# if we    have      n*pi   +  xi*pi,  stntme  = {0: [n, pi], 1: [xi, pi]}
# a list containing either one hadron, or two hadrons,  or  three  hadrons
# here      we      do      not      have     a   superposition of hadrons
# newhadrons=nwhdrns is a list of hadrons existing within time  and number
# counter is needed to   give    the     number     of     the     hadrons
# if the hadron is on source, i.e. tm = 0, then we numbering is   reversed
# A list containing all   hdrn/h2 or h3   objects   of the   superposition
# at the end, sprpstncntr denotes the numbe r of  superpositions  we  have
# st.gAhdrn() is a list containing either 1 hadron, 2 hadrons or 3 hadrons
# we take now each hadron of the type hdrn and make out  of  it  a  hadron
# that lives in  a   specific    time   slice,  i.e. an OpTimeSlice-object
# in this methods we rewrite the hadrons in  stntme  in  terms  of  quarks
# that lives in  a   specific    time   slice,  i.e. an OpTimeSlice-object
# in sp0 we have the map of      hadrons,       where       each       key  
# corresponds            to    a  superposition, i.e. to a line of hadrons
# each term in line_of_hadrons corresponds to a hadron, i.e. a hdrn-object
# we take each hadron in the line, we start with 0, its quarks will     be
# numbered as q1, q2, q3-> 0 + 1, 0 +2, 0+3. For the next hadron we   need
# to know   how    many    quarks     we     already     have      behind!
# we increase fnl_qrk_nmbr by the number of the quarks   aleady    behind!
# make   a  list of the quarks and add it as a quark line to the final map
# quarks_on_time[vp0] corresponds to the hadrons   in   superposition-term
# of number "vp0"     written     in     terms     of     their     quarks
# at the end we obtain a map of a superposition of quarks, which  will  be
# later contracted with each others/other sets of hadrons on  time  slices
# the next step is   naturally  to organize the quarks in "quarks_on_time"
# according to their flavor and wheather they are in bar  or  not  in  bar
# this is taken care     of        in the        class        organizeq_qb
# at the end, the map spf contains all superpositions,  each superposition
# is splitted into a map corresponding  to  the  flavor  structure  of  it
# this loop is to extract the existing flavors in each line   of    quarks
# now we insert into each existing flavor all quarks corresponding  to  it
# now we give a list of containing the map spf and the  number  of  quarks
# nr_of_qrks is needed to give the correct ordering of quarks in the total
# operator, whose quarks are to be contracted     with     each     others
# define a product between operators living on time slices. This is needed
# since the operator which we want to contract is made  out  of  operators
# living on sink, source and/or in currents. That is we need a  symstamtic
# product rule between these operators on time. What is important here  is
# the ordering, the number grassmann must always be correctly obtained and
# forwarded. This        is         done          as              follows:
# self * other-> number of quarks from self must  be  forwarded  to  other
# qrks_on_time1 = [map of organized flavors, number of existing quarks nr]
# now       we          forward           nr              to         other
# In general, two hadron states living in time   will   always  produce  a
# wicktract-type object. In case we have three states multiplied, or  more
# O1 * O2 * O3 * O4 * ... = wicktract(O1, O2) * O3 * O4 = wicktract(O1,..)
# sp0 is the last set in wicktract containing the map and length of quarks
# qrks_on_time1       is       the    total   list  of  maps  and  lengths
# qrks_on_time2 is the list of map and length of the  self  but  with  the
# grassmann length nr    needed    to  give the quarks in correct ordering
# qrks_on_time  is    the    new    input    of    the    wicktract-object
class OpTimeSlice:
    def __init__(self, tm, hdrn_state):
        self.tm         = tm
        self.hdrn_state = hdrn_state
    def organizeH(self):
        stntme = {}
        if isinstance(self.hdrn_state, (hdrn, h2, h3, h4)):
            statstic_hadrons0 = (self.hdrn_state).gAhdrn()
            nwhdrns = []
            counter = 0
            if self.tm == 0:
                statstic_hadrons = reversed(statstic_hadrons0)
            else:
                statstic_hadrons = statstic_hadrons0
            for hadron in statstic_hadrons:
                nwhdrns.append(hdrn_ontime(self.tm, counter, hadron))
                counter += 1
            stntme[0] = nwhdrns
        elif isinstance(self.hdrn_state, (mhdrn, mh2, mh3, mh4)):
            superposition = (self.hdrn_state).gAhdrn()
            sprpstncntr = 0
            for st in superposition:
                statstic_hadrons0 = st.gAhdrn()
                nwhdrns = []
                counter = 0
                if self.tm == 0:
                    statstic_hadrons = reversed(statstic_hadrons0)
                else:
                    statstic_hadrons = statstic_hadrons0
                for hadron in statstic_hadrons:
                    nwhdrns.append(hdrn_ontime(self.tm, counter , hadron))
                    counter += 1
                stntme[sprpstncntr] = nwhdrns
                sprpstncntr += 1
        else:
            raise TypeError(f"Undefined type of states")
        return stntme
    def organizeQ(self, grassmann):
        quarks_on_time   = {}
        number_of_quarks = []
        sp0 = self.organizeH()
        for vp0 in sp0:
            line_of_hadrons = sp0[vp0]
            line_of_quarks = []
            for hadron in line_of_hadrons:
                quarks = hadron.gqrks(grassmann + len(line_of_quarks))
                line_of_quarks.extend(quarks)
            quarks_on_time[vp0] = line_of_quarks
            #This has been modified
            number_of_quarks.append(grassmann + len(line_of_quarks))
        for nr1 in number_of_quarks:
            for nr2 in number_of_quarks:
                if nr1 != nr2:
                    print("You have a superposition of quarks")
                    print(" with different number of quarks in each term")
                    raise TypeError(f"Error!")
        return [quarks_on_time, number_of_quarks[0]]
    def organizeq_qb(self, grassmann):
        sp1 = self.organizeQ(grassmann)[0]
        nr_of_qrks = self.organizeQ(grassmann)[1]
        spf = {}
        counter = 0
        for line_of_quarks in sp1:
            flavor = {}
            for quark in sp1[line_of_quarks]:
                flavor[quark.gflvr()] = []
            for quark in sp1[line_of_quarks]:
                flavor[quark.gflvr()].append(quark)
            spf[counter] = flavor
            counter += 1
        return [spf, nr_of_qrks]
    def Laudtracto(self):
        qrks_on_time1 = self.organizeq_qb(0)
        return wicktract(qrks_on_time1).Laudtracto()
    def __mul__(self, other):
        if isinstance(other, OpTimeSlice):
            qrks_on_time1 = self.organizeq_qb(0)
            nr            = self.organizeq_qb(0)[1]
            qrks_on_time2 = other.organizeq_qb(nr)
            return wicktract(*(qrks_on_time1, qrks_on_time2))
    def __rmul__(self, other):
        if isinstance(other, wicktract):
            sp0 = other.gqrks_on_time()[-1]
            nr  = sp0[1]
            #this has beedn modified
            qrks_on_time1 = other.gqrks_on_time()
            #.gqrks_on_time()->list(*qrks_on_time)
            qrks_on_time2 = self.organizeq_qb(nr)
            #.organizeq_qb(self, grassmann)->list(spf, nr_of_qrks)
            # but qrks_on_time1 is of the form list(list(spf, nr_of_qrks))
            qrks_on_time  = qrks_on_time1 + [qrks_on_time2]
            return wicktract(*qrks_on_time)
        else:
            raise TypeError(f"Error: Undefined product with quarks living in time!")

#********
#Hier you need to take into the account, that in case you exchange 2 pairs, then if these tow pairs are contracted with each others, then you do not have the options to do the exchangements in one single propagator!
#********
def quark_exchanger(quark, ex_quark1, ex_quark2):
    factor = quark.gff()
    if quark == ex_quark1:
        return [True, ex_quark2]
    elif quark == ex_quark2:
        return [True, ex_quark1]
    else:
        return [False, quark]
def DiagramRewriter(diagram, prslst):
    alprp = diagram.gpropagators()
    nwprp = []
    for propagator in alprp:
        prpfactor        = propagator.gff()
        quark1           = propagator.gqrk1()
        quark2           = propagator.gqrk2()
        #exchange first quark in the propagator
        for exchanging_pair in prslst:
            ex_q1     = exchanging_pair.sq1()
            ex_q2     = exchanging_pair.sq2()
            exchanged = quark_exchanger(quark1, ex_q1, ex_q2)
            quark1_exchanged = exchanged[1]
            if exchanged[0]:
                break
        #exchange second quark in the propagator
        for exchanging_pair in prslst:
            ex_q1     = exchanging_pair.sq1()
            ex_q2     = exchanging_pair.sq2()
            exchanged = quark_exchanger(quark2, ex_q1, ex_q2)
            quark2_exchanged = exchanged[1]#hier bin ich
            if exchanged[0]:
                break
        nwprp.append(prpgtr(prpfactor, quark1_exchanged, quark2_exchanged))
    factor = diagram.gff()
    for exchange in prslst:
        factor *= exchange.gsign()
    return dgrm(factor, *nwprp)
class pair_and_sign:
    def __init__(self, quark1, quark2):
        self.quark1 = quark1
        self.quark2 = quark2
    def sq1(self):
        return self.quark1
    def sq2(self):
        return self.quark2
    def gsign(self):
        if self.quark1.ghdrn_t() != self.quark2.ghdrn_t():
            raise TypeError("Failed to extract simplification pairs 2")
        else:
            if self.quark1.ghdrn_t() == 'lambda' or self.quark1.ghdrn_t() == 'lambdaB':
                return 1
            else:
                return -1
#********
# as explained above, qrks_on_time = [flavor map, length    of the quarks]
# qrks_on_time  is    the    new    input    of    the    wicktract-object
# as explained above, a wicktract-object can appear only on  the left-hand
# side of a product, so we will not need  a      def __rmul__(self, other)
# sp0 is the last set in wicktract containing the map and length of quarks
# qrks_on_time1       is       the    total   list  of  maps  and  lengths
# qrks_on_time2 is the list of map and length of the  self  but  with  the
# grassmann length nr    needed    to  give the quarks in correct ordering
# qrks_on_time  is    the    new    input    of    the    wicktract-object
# extract_maps has the output:[  {0: {flvrs:[], ...}, 1: {flvrs:[], ...}},
#                      {0: {flvrs:[], ...}, 1: {flvrs:[], ...}}, ...     ]
# the maps in the output of extract_maps are multiplied with each   others
# in this method we perform the multiplication and create a  superposition
# of multiplied maps with each others. First we define the       following
# line1 and line2 are   quark   lines  (from above)  that  are  multiplied
# with                           each                               others
# i.e. line1 is of the form {'flavor_1':[quarks], 'flavor_2':[quarks],...}
# map1 and map2 are maps containing quark lines (from above) i.e. mapi is
# of the form {0: {'flavor_1':[quarks], 'flavor_2':[quarks],...}        ,
#             1: {'flavor_3':[quarks], 'flavor_4':[quarks],...}         ,
#                                                              .........}
# the final result of maps_lines_multiplier is a map, that contains a map
# coming from multiplying two maps with each others. The length of   this
# final map is equal to the product of the lengths of the two input  maps
# multiply              the           first            two           maps:
# multiply             the        first      two  maps with the last  one:
# multiply the first two and the last   two   maps   with   each   others:
class wicktract:
    def __init__(self, *qrks_on_time):
        self.qrks_on_time = qrks_on_time
    def gqrks_on_time(self):
        return list(self.qrks_on_time)
    def squark_pair(self):
        def generate_combinations(pairs):
            result = []
            N = len(pairs)
            for r in range(1, N + 1):
                result.extend(combinations(pairs, r))
            comList = [list(comb) for comb in result]
            print(f"{len(comList)} combinations have been generated to do simplifications due to Collin's coefficients")
            print("Perform now the final simplification...")
            print("")
            return comList
        p_sp0 = ['Sigma', 'SigmaB', 'xi', 'xiB', 'nucleon', 'nucleonB', 'lambda', 'lambdaB']
        seen  = []
        f_sp0 = ['delta', 'deltaB', 'omega', 'omegaB']
        f_sp  = {}
        p_sp  = {}
        for operator in self.gqrks_on_time():
            for flavor in operator[0][0]:
                for quark in operator[0][0][flavor]:
                    if quark.ghdrn_t() in f_sp0 and quark not in seen:
                        seen.append(quark)
                        if quark.ghdrn_t() not in f_sp:
                            f_sp[quark.ghdrn_t()] = {}
                        if quark.ghdrn_n() not in f_sp[quark.ghdrn_t()]:
                            f_sp[quark.ghdrn_t()][quark.ghdrn_n()] = [quark]
                        else:
                            f_sp[quark.ghdrn_t()][quark.ghdrn_n()].append(quark)
        exchange_pairs = []
        if len(f_sp) != 0:
            for hadron_type in f_sp:
                for hadron_number in f_sp[hadron_type]:
                    if len(f_sp[hadron_type][hadron_number]) != 3:
                        raise TypeError(f"Failed to extract simplification pairs for {hadron_type} number {hadron_number}")
                    else:
                        q0 = f_sp[hadron_type][hadron_number][0]
                        q1 = f_sp[hadron_type][hadron_number][1]
                        q2 = f_sp[hadron_type][hadron_number][2]
                        exchange_pairs.extend([pair_and_sign(q0, q1), pair_and_sign(q0, q2), pair_and_sign(q1, q2)])
        for operator in self.gqrks_on_time():
            for flavor in operator[0][0]:
                for quark in operator[0][0][flavor]:
                    if (quark.ghdrn_t() in p_sp0) and (quark not in seen) and (quark.gqrk_hdrn_p() != 2):
                        seen.append(quark)
                        if quark.ghdrn_t() not in p_sp:
                            p_sp[quark.ghdrn_t()] = {}
                        if quark.ghdrn_n() not in p_sp[quark.ghdrn_t()]:
                            p_sp[quark.ghdrn_t()][quark.ghdrn_n()] = [quark]
                        else:
                            p_sp[quark.ghdrn_t()][quark.ghdrn_n()].append(quark)
        if len(p_sp) != 0:
            for hadron_type in p_sp:
                for hadron_number in p_sp[hadron_type]:
                    if len(p_sp[hadron_type][hadron_number]) != 2:
                        raise TypeError(f"Failed to extract simplification pairs for {hadron_type} number {hadron_number}")
                    else:
                        q0 = p_sp[hadron_type][hadron_number][0]
                        q1 = p_sp[hadron_type][hadron_number][1]
                        exchange_pairs.append(pair_and_sign(q0, q1))
        #hier am 23.03 um 20:07 gestoppt
        return generate_combinations(exchange_pairs)
    def __mul__(self, other):
        if isinstance(other, OpTimeSlice):
            sp0 = self.gqrks_on_time()[-1]
            nr  = sp0[1]
            #this has been modified
            qrks_on_time1 = self.gqrks_on_time()
            #.gqrks_on_time()->list(*qrks_on_time)
            qrks_on_time2 = other.organizeq_qb(nr)
            #.organizeq_qb(self, grassmann)->list(spf, nr_of_qrks)
            # but qrks_on_time1 is of the form list(list(spf, nr_of_qrks))
            qrks_on_time  = qrks_on_time1 + [qrks_on_time2]
            return wicktract(*qrks_on_time)            
        else:
            raise TypeError(f"Error: Undefined product with quarks living in time!")
    def extract_maps(self):
        list_of_maps = []
        for map_number in self.gqrks_on_time():
            list_of_maps.append(map_number[0])
        return list_of_maps
    def super_position_of_maps(self):
        def quark_lines_multiplier(line1, line2):
            multiplied_quark_line = {}
            for flavor1 in line1:
                if flavor1 in line2:
                    multiplied_quark_line[flavor1] = line1[flavor1] + line2[flavor1]
                else:
                    multiplied_quark_line[flavor1] = line1[flavor1]
            for flavor2 in line2:
                if flavor2 not in multiplied_quark_line:
                    multiplied_quark_line[flavor2] = line2[flavor2]
            return multiplied_quark_line
        def maps_lines_multiplier(map1, map2):
            fnl_mp = {}
            counter   = 0
            for nmbr1 in map1:
                for nmbr2 in map2:
                    fnl_mp[counter] = quark_lines_multiplier(map1[nmbr1], map2[nmbr2])
                    counter += 1
            return fnl_mp
        sp0   = self.extract_maps()
        karl  = len(sp0)
        if karl == 1:
            return sp0[0]
        elif karl == 2:
            return maps_lines_multiplier(sp0[0], sp0[1])
        elif karl == 3:
            frstwmps = maps_lines_multiplier(sp0[0], sp0[1])
            return maps_lines_multiplier(frstwmps, sp0[2])
        elif karl == 4:
            frstwmps = maps_lines_multiplier(sp0[0], sp0[1])
            lstwmps = maps_lines_multiplier(sp0[2], sp0[3])
            return maps_lines_multiplier(frstwmps,lstwmps)
        elif karl == 5:
            frstwmps = maps_lines_multiplier(sp0[0], sp0[1])
            nxtwmps = maps_lines_multiplier(sp0[2], sp0[3])
            flmap = maps_lines_multiplier(frstwmps,nxtwmps)
            return maps_lines_multiplier(flmap, sp0[4])
        else:
            raise TypeError("Error: Current version handels only up to five operators living in different time slices")

    def final_super_position_of_maps(self):
# filter now the list, to see if the result might be zero or not, in case
# not enough quarks   can     be      contracted   with    each    others
        sp0 = self.super_position_of_maps()
        delete_elementes = []
        nmbrfsprpstn = [nr for nr in sp0]
        for i in nmbrfsprpstn:
            to_break = False
            for flvr in sp0[i]:
                if to_break:
                    break
                if flvr[-1] == 'B':
                    if flvr[0] not in sp0[i]:
                        delete_elementes.append(i)
                        to_break = True
                        continue
                    else:
                        if len(sp0[i][flvr[0]]) != len(sp0[i][flvr]):
                            delete_elementes.append(i)
                            to_break = True
                            continue
                else:
                    nflvr = flvr + 'B'
                    if nflvr not in sp0[i]:
                        delete_elementes.append(i)
                        to_break = True
                        continue
                    else:
                        if len(sp0[i][flvr]) != len(sp0[i][nflvr]):
                            delete_elementes.append(i)
                            to_break = True
                            continue
        for i in list(set(delete_elementes)):
            del sp0[i]
        return sp0
    def Laudtracto(self):
        sp1 = self.final_super_position_of_maps()
        final_result = []
        print(f"In total there are {len(sp1)} Correlator/s")
        print(" ")
        c_counter = 1
        for sr_position in sp1:
            print(f"****** Obtain Diagrams for Correlator {c_counter} ******")
            final_result.append(creatograms(sp1[sr_position]))
            c_counter += 1
        print("**********************************************")
        if len(sp1) > 1:
            print("Now add all diagrams of different correlators to each other")
            print(" ")
            total_diagrams = []
            for dgr_container in final_result:
                total_diagrams.extend(dgr_container.gdiagrams())
            print(f"All diagrams are added into one correlator. There are {len(total_diagrams)} diagrams")
            #sp1 = dgrm_container(*total_diagrams).gdgrms()
            sp1 = dgrm_container(*total_diagrams)
            do_extra_simplification = True
            print(" ")
            print("**********************************************")
        elif len(sp1) == 1:
            print(" ")
            total_diagrams = []
            for dgr_container in final_result:
                total_diagrams.extend(dgr_container.gdiagrams())
            #sp1 = dgrm_container(*total_diagrams).gdiagrams()
            sp1 = dgrm_container(*total_diagrams)
            do_extra_simplification = False
            print(" ")
            print("********")
        else:
            raise TypeError("Failed to identity correlators")
        extract_colbaryons_pairs = self.squark_pair()
        if len(extract_colbaryons_pairs) != 0:
            sp1 = sp1.gdgrms() if do_extra_simplification else sp1.gdiagrams()
            simsp1   = []
            seen     = set()
            counter  = 0
            for i1 in range(len(sp1)):
                if i1 in seen:
                    continue
                else:
                    simsp1.append(sp1[i1])
                for i2 in range(i1 + 1, len(sp1)):
                    if i2 in seen:
                        continue
                    for exchanging_pair_list in extract_colbaryons_pairs:
                        rewritten_diagram = DiagramRewriter(sp1[i2], exchanging_pair_list)
                        if sp1[i1] == rewritten_diagram:
                            print(f"Diagram {i2} has been added to diagram {i1}")
                            counter += 1
                            simsp1[-1] = simsp1[-1] + rewritten_diagram
                            seen.add(i2)
                            break
                if i1 % 10 == 0 and i1 != 0:
                    print(f"****** Simplifying went through diagram {i1} ******")
            print(f" {counter} diagrams have been reduced")
            sp1 = dgrm_container(*simsp1).gdgrms() if do_extra_simplification else dgrm_container(*simsp1).gdiagrams()
        else:
            sp1 = sp1.gdgrms() if do_extra_simplification else sp1.gdiagrams()
        print("To visualize the diagrams use the attribute bulavision() ")
        return sp1
class qrk_ontime:
# a class which takes quarks  and  associates  time  and  number  to  them
# by time is meant  sink, source or current/s, while the number stands for
# their position in the contraction  list,  this  position  is  needed  to
# sepcify    the     sign      coming      from the      Grassmann-Algebra
# hdrn_n stands for the number    of    the    hadron    they    come from
    def __init__(self, hdrn_ontime, nmbr, quark):
        self.hdrn_ontime = hdrn_ontime
        self.tm          = self.hdrn_ontime.gtm()
        self.nmbr        = nmbr
        self.hdrn_n      = self.hdrn_ontime.gnmbr()
        self.quark       = quark
# tm represents the time, nmbr for the  position  in  the   final operator
# hdrn_n represents the number of the hadron   in    the    final   result
# quark is a qrk-typed object, from which many  properties  are  extracted
    def gtm(self):
        return self.tm
    def gnmbr(self):
        return self.nmbr
    def ghdrn_n(self):
        return self.hdrn_n
    def gquark(self):
        return self.quark
    def gff(self):
        return (self.gquark()).gff()
    def ghdrn_t(self):
        return (self.gquark()).ghdrn_t()
    def gmntm(self):
        return (self.gquark()).gmntm()
    def gflvr(self):
        return (self.gquark()).gflvr()
    def gqrk_hdrn_p(self):
        return (self.gquark()).gqrk_hdrn_p()
    def ghdrn_ontime(self):
        return self.hdrn_ontime
    def gqrktprp(self):
        return (self.gtm(), self.ghdrn_n(), self.gnmbr()) + self.quark.gqrkprp()
    def __eq__(self, other):
        return isinstance(other, qrk_ontime) and (self.gqrktprp() == other.gqrktprp())
class hdrn_ontime:
# a class which takes hadrons and  associates  time  and  nmbr    to  them
# the time stands for sink, source or current, while the nmbr   stands for
# their position in the contraction  list,  this  position  is  needed  to
# sepcify the sign of the diagram  due    to    the      Grassmann-Algebra
    def __init__(self, tm, nmbr, hadron):
        self.tm     = tm
        self.nmbr   = nmbr
        self.hadron = hadron
    def gtm(self):
        return self.tm
    def gnmbr(self):
        return self.nmbr
    def ghdrn(self):
        return self.hadron
    def ghdrn_t(self):
        return self.hadron.ghdrn_t()
    def gnqrks(self):
# this is needed to determine the correct sign coming from the   grassmann
        return len((self.hadron).O())
    def gqrks(self, counter):
# this generates the version    of     the     quarks    within the hadron
# that        exists           on          the          time     slice  tm
        qrk_list = []
        for i, quark in enumerate((self.hadron).O()):
            qrk_list.append(qrk_ontime(self, counter + i, quark))
        return qrk_list
    def __eq__(self, other):
        return isinstance(other, hdrn_ontime) and self.gtm() == other.gtm() and self.ghdrn() == other.ghdrn()
### Untill here all seems logic

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc


def diagramplot(diagram):
    all_propagators = diagram.gpropagators()
    seen       = set()
    lst_ts     = set()
    lst_hdrns  = []
    for propagator in all_propagators:
        if (propagator.ghdrn1().gnmbr(), propagator.ghdrn1().ghdrn_t(), propagator.ghdrn1().gtm()  ) not in seen:
            lst_hdrns.append(propagator.ghdrn1())
            seen.add((propagator.ghdrn1().gnmbr(), propagator.ghdrn1().ghdrn_t(), propagator.ghdrn1().gtm() ))
            lst_ts.add(propagator.ghdrn1().gtm())
        if (propagator.ghdrn2().gnmbr(), propagator.ghdrn2().ghdrn_t(), propagator.ghdrn2().gtm() )not in seen:
            lst_hdrns.append(propagator.ghdrn2())
            seen.add((propagator.ghdrn2().gnmbr(), propagator.ghdrn2().ghdrn_t(), propagator.ghdrn2().gtm() ))
            lst_ts.add(propagator.ghdrn2().gtm())
    #maximal number of hadrons on a time slice
    mnhts   = {}
    hdrns_t = {}
    for hadron in lst_hdrns:
        mnhts[hadron.gtm()]   = [0]
        hdrns_t[hadron.gtm()] = []
    for hadron in lst_hdrns:
        mnhts[hadron.gtm()][0] += 1
    hadron_ts = 0
    for timslice in mnhts:
        if mnhts[timslice][0] > hadron_ts:
            hadron_ts = mnhts[timslice][0]
    for hadron in lst_hdrns:
        hdrns_t[hadron.gtm()].append(hadron)
    #Build now the coordinate system with the Hadrons and quarks inside of them
    fig, ax     = plt.subplots()
    time_slices = len(lst_ts)
    sky_blue = (135/255, 206/255, 235/255)
    T        = {}
####
    if len(hdrns_t) == 1:
        # Determine which timeslice we have
        timeslice_key = list(hdrns_t.keys())[0]
        y_ver = {timeslice_key: 1} 
        if timeslice_key == 0:
            ax.text(1, 3 * hadron_ts + 4.5, 'src', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        elif timeslice_key == 1:
            ax.text(1, 3 * hadron_ts + 4.5, 'snk', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        else:
            raise TypeError("Single time slice must be either source (0) or sink (1)")  
        for timslice in hdrns_t:
            for hadron in hdrns_t[timslice]:
                yco = 3 * hadron_ts + 2 * y_ver[hadron.gtm()]
                xco = 1
                hadronP = patches.Ellipse((xco, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                ax.text(xco + 1, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                ax.text(xco + 1, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                for i, quark in enumerate(hadron.ghdrn().O()):
                    ax.text(xco, yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                    T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (xco, yco + 1 - i)
                y_ver[hadron.gtm()] -= 2
                ax.add_patch(hadronP)
####
    if len(hdrns_t) == 2:
        y_ver = {0 : 1, 1: 1}
        ax.text(1, 3 * hadron_ts + 4.5, 'snk', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        ax.text(3 * time_slices, 3 * hadron_ts + 4.5, 'src', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        for timslice in hdrns_t:
            for hadron in hdrns_t[timslice]:
                yco = 3 * hadron_ts + 2 * y_ver[hadron.gtm()]
                if timslice == 1:
                    hadronP = patches.Ellipse((1, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(0, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(0, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(1 , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (1 , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                elif timslice == 0:
                    xco = 3 * time_slices
                    hadronP = patches.Ellipse((xco, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(xco + 1, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(xco + 1, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(xco , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (xco , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                else:
                    raise TypeError("Two time slices are given, but they do not correspond to sink and source (1, 0)")
    elif len(hdrns_t) == 3:
        y_ver = {0 : 1, 1: 1, 2: 1}
        ax.text(1, 3 * hadron_ts + 4.5, 'snk', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        ax.text(3 * time_slices, 3 * hadron_ts + 4.5, 'src', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        ax.text((3 * time_slices - 1)/2 + 1, 3 * hadron_ts + 4.5, 'J (2)', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        for timslice in hdrns_t:
            for hadron in hdrns_t[timslice]:
                yco = 3 * hadron_ts + 2 * y_ver[hadron.gtm()]
                if timslice == 1:
                    hadronP = patches.Ellipse((1, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(0, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(0, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(1 , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (1 , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                elif timslice == 2:
                    xco = (3*time_slices-1)/2 + 1
                    hadronP = patches.Ellipse((xco, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(xco-1, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(xco-1, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(xco , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (xco , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                elif timslice == 0:
                    xco = 3 * time_slices
                    hadronP = patches.Ellipse((xco, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(xco + 1.2, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(xco + 1, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(xco , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (xco , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                else:
                    raise TypeError("There must be at least one time slice for  sink (1), one source (0) and a current (2)")
    elif len(hdrns_t) == 4:
        y_ver = {0 : 1, 1: 1, 2: 1, 3: 1}
        ax.text(1, 3 * hadron_ts + 4.5, 'snk', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        ax.text(3 * time_slices, 3 * hadron_ts + 4.5, 'src', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        ax.text((3 * time_slices - 1)/2 - 1, 3 * hadron_ts + 4.5, 'J (2)', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        ax.text((3 * time_slices - 1)/2 + 3, 3 * hadron_ts + 4.5, 'J (3)', fontsize=15, ha='center', va='center', color='gray', weight='bold')
        for timslice in hdrns_t:
            for hadron in hdrns_t[timslice]:
                yco = 3 * hadron_ts + 2 * y_ver[hadron.gtm()]
                if timslice == 1:
                    hadronP = patches.Ellipse((1, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(-0.4, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(-0.4, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(1 , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (1 , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                elif timslice == 2:
                    xco = (3*time_slices-1)/2-1
                    hadronP = patches.Ellipse((xco, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(xco-1.2, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(xco-1.2, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(xco , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (xco , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                elif timslice == 3:
                    xco = (3*time_slices-1)/2+3
                    hadronP = patches.Ellipse((xco, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(xco-1.2, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(xco-1.2, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(xco , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (xco , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                elif timslice == 0:
                    xco = 3 * time_slices
                    hadronP = patches.Ellipse((xco, yco), width=1, height=3, edgecolor='black', facecolor=sky_blue, lw=2)
                    ax.text(xco + 1.5, yco+1, hadron.ghdrn_t(), fontsize=10, ha='center', va='center', color='black')
                    ax.text(xco + 1.5, yco+0.5, f"({hadron.gnmbr()})", fontsize=10, ha='center', va='center', color='black')
                    for i, quark in enumerate(hadron.ghdrn().O()):
                        ax.text(xco , yco + 1 - i, quark.gqrk_hdrn_p(), fontsize=14, ha='center', va='center', color='black')
                        T[(hadron.gtm(), quark.ghdrn_t(), hadron.gnmbr(), quark.gqrk_hdrn_p())] = (xco , yco + 1 - i)
                    y_ver[hadron.gtm()] -= 2
                    ax.add_patch(hadronP)
                else:
                    raise TypeError("There must be at least one time slice for  sink (1), one source (0) and current2 (2 and 3) ")
    for propagator in all_propagators:
        coordinate_quarkBar = T[(propagator.gHbar().gtm(), propagator.gbar().ghdrn_t(), propagator.gHbar().gnmbr(), propagator.gbar().gqrk_hdrn_p())]
        coordinate_quarknBar = T[(propagator.gHnbar().gtm(), propagator.gnbar().ghdrn_t(), propagator.gHnbar().gnmbr(), propagator.gnbar().gqrk_hdrn_p())]
        x1 = coordinate_quarkBar[0]
        y1 = coordinate_quarkBar[1]
        x2 = coordinate_quarknBar[0]
        y2 = coordinate_quarknBar[1]
        if propagator.gbar().gflvr() in ['u', 'uB', 'd', 'dB']:
            clr = 'black'
        elif propagator.gbar().gflvr() in ['s', 'sB']:
            clr = 'red'
        elif propagator.gbar().gflvr() in ['c', 'cB']:
            clr = 'green'
        elif propagator.gbar().gflvr() in ['b', 'bB']:
            clr = 'orange'
        else:
            raise TypeError("Quatk flavor mus be in [u, d, s, c, uB, dB, sB, cB]")
        if propagator.gHbar().gtm() != propagator.gHnbar().gtm():
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color=clr, lw=1.5, mutation_scale=15)
            ax.add_patch(arrow)
        else:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),connectionstyle="arc3,rad=0.5",arrowstyle="->", color=clr, lw=1.5, mutation_scale=15)
            ax.add_patch(arrow)
    ax.set_xlim(0, 3 * time_slices +1)
    ax.set_ylim(0, 3 * hadron_ts + 4)
    ax.axis('off')
    plt.show()