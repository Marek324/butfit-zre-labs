import numpy as np
from .frame import frame

def lag_acf(s,lseg,rseg,Lmin,Lmax,thr):
    """
    Lacf = lag_acf (s,lseg,rseg,Lmin,Lmax,thr)
    
     estimation of lags by autocorrelation. 
     Lacf - computed lags in a row vector - lag in samples in case frame is voiced, 
       zero otherwise. 
     s - signal in a row vector. 
     lseg - length of window 
     rseg - overlap of window 
     Lmin - minimum lag, for 8000 Hz use 20. 
     Lmax - maximum lag, for 8000 Hz and 160 sample frames, use something like 150. 
     thr - threshold to determine voiceness - in case Rmax > thr * R[0], voiced
                                                      Rmax <= thr * R[0], unvoiced
       reasonable value is 0.3 
    """
    sr = frame (s,lseg,rseg); 
    Nram = sr.shape[0]
    Lacf = np.zeros(Nram)
    for n in range(Nram):
        x = sr[n] 
        R = np.correlate(x, x, mode='full')[lseg-1:]
        Rmaxarg = np.argmax(R[Lmin:Lmax])
        Rmax = R[Lmin + Rmaxarg]
        if Rmax >= thr*R[0]:
            L = Lmin + Rmaxarg
        else:
            L = 0
        Lacf[n] = L; 
    return Lacf
    
def lag_nccf(s,lseg,rseg,Lmin,Lmax,thr): 
    """
     function Lnccf = lag_nccf (s,lseg,rseg,Lmin,Lmax,thr); 
    
     estimation of lags by normalized cross-correlation 
     Lnccf - computed lags in a row vector - lag in samples in case frame is voiced, 
       zero otherwise. 
     s - signal in a vector. 
     lseg - length of window 
     rseg - overlap of window 
     Lmin - minimum lag, for 8000 Hz use 20. 
     Lmax - maximum lag, for 8000 Hz and 160 sample frames, use something like 150. 
     thr - threshold to determine voiceness - in case Rmax > thr * R[0], voiced
                                                      Rmax <= thr * R[0], unvoiced
       reasonable value is 0.7 
    """
    sr = frame (s,lseg,rseg)
    Nfr = sr.shape[0];  # just for fun as I am too lazy to retype the formula from 
    # lectures, these frames will be never used ... 

    # prepare a version of s from which the shifted signal will be taken. Can have problems
    # at the beginning, so pre-prend Lmax zeros at the beginning. 
    # this signal will need special indexing, with an index always += Lmax. 
    saux = np.pad(s, (Lmax, 0))

    Lnccf = np.zeros(Nfr)
    start = 0
    to = start + lseg
    for fr in range(Nfr):
        selected = s[start:to]  # nonshifted frame
        E1 = np.sum(selected**2) # energy of non-shifted frame.  

        Rnccf = np.zeros(Lmax + 1)
        for n in range(Lmax):
            starts = start - n + Lmax
            tos = to - n + Lmax  # indexes of the shifted frame
                                 # Lmax is there because of the zeros
                                 # added at the beginning !
            shifted = saux[starts:tos] 
            E2 = np.sum(shifted**2)  # energy of the shifted one
            numerator = np.sum(selected * shifted)
            nccf = numerator / np.sqrt(E1 * E2)
            Rnccf[n] = nccf
        
        Rmaxarg = np.argmax(Rnccf[Lmin:Lmax])
        Rmax = Rnccf[Lmin + Rmaxarg]
        if Rmax >= thr*Rnccf[0]:
            L = Lmin + Rmaxarg
        else:
            L = 0
        Lnccf[fr] = L
        start += lseg - rseg
        to = start + lseg
    return Lnccf
