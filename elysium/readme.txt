The 3N3N.hdf5 file was generated using the following commands:
O1 = Nucleon(1/2) * Nucleon(1/2) * Nucleon(1/2)
O1_ontime = OpTimeSlice(1, O1)
O2 = bar(O1)
O2_ontime = OpTimeSlice(0, O2)
Correlator = O1_ontime * O2_ontime
Result = Correlator.Laudtracto()
writeresults(Result, O1_ontime, O2_ontime, path = 'elysium/3N3N.hdf5')
