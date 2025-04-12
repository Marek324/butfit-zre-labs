import numpy as np
import scipy.signal as sg
import scipy
import librosa
from .frame import frame
from .lag import lag_nccf

def mylpc(x, P):
    R = np.array([np.sum(x[i:]*x[:len(x)-i]) for i in range(P+1)])
    return np.hstack([[1], np.linalg.inv(scipy.linalg.toeplitz(R[:-1])) @ -R[1:]])

def syntnoise(A,G,P,lram):
    Nram = len(G)
    init=np.zeros(P)
    ss = []
    for i in range(Nram):
        a = np.hstack([[1], A[i]])
        g = G[i] 
        excit = np.random.normal(size=lram)
        synt, final = sg.lfilter([g], a, excit, zi=init)
        init = final
        ss.append(synt)
    return np.hstack(ss)
    

def synthesize(A,G,L,P,lram):
    Nram = len(G) 
    init=np.zeros(P)        # initial conditions of the filter
    ss = np.zeros(Nram * lram) # preallocation is needed for long signals. 

    # some initial values - position of the next pulse for voiced frames
    nextvoiced = 0

    f = 0
    to = f + lram
    for n in range(Nram):
      a = np.hstack([1, A[n]]) # prepending with 1 for filtering
      g = G[n]
      l = L[n] 
      
      # in case the frame is unvoiced, generate noise
      if l == 0: 
        excit = np.random.normal(size=lram)
      else: # if it is voiced, generate some pulses
        where = np.arange(nextvoiced, lram, l, dtype= int)
        # ok, this is for the current frame, but where should be the 1st pulse in the 
        # next one ? 
        nextvoiced = np.max(where) + l - lram
        # generate the pulses
        excit = np.zeros(lram)
        excit[where] = 1
        
      # and set the power of excitation  to one - no necessary for noise, but anyway ...
      power = np.sum(excit ** 2) / lram
      excit = excit / np.sqrt(power)
      # check 
      # print(np.sum(excit ** 2) / lram)
      
      # now just generate the output  
      synt, final = sg.lfilter([g], a, excit,zi=init)
      ss[f:to] = synt # !!! this line was originally at the end. 
      init = final 
      f = f + lram
      to = f + lram 
    return ss


def param(wavname,lram,pram,P, Fs=8000):
    s = librosa.load(wavname, sr=Fs)[0]
    sm = s - np.mean(s); 
    sr = frame(sm, lram, pram)

    Nram = sr.shape[0]
    A = np.zeros((Nram,P))
    G = np.zeros(Nram) 
    for i in range(Nram):
        a = mylpc(sr[i], P)
        e = sg.lfilter(a, [1], sr[i])
        A[i] = a[1:]
        G[i] = np.sqrt(np.sum(e**2))
    return A, G, Nram


def param2(wavname, lram, pram, P, Lmin, Lmax, thr, Fs=8000):
    if type(wavname) is str:
        s = librosa.load(wavname, sr=Fs)[0]
        sm = s - np.mean(s); 
    else:
        sm = wavname
    sr = frame(sm, lram, pram)

    Nram = sr.shape[0]
    A = np.zeros((Nram,P))
    G = np.zeros(Nram) 
    for i in range(Nram):
        a = mylpc(sr[i], P)
        e = sg.lfilter(a, [1], sr[i])
        A[i] = a[1:]
        G[i] = np.sqrt(np.sum(e**2))
    Lnccf = lag_nccf(sm, lram, pram, Lmin, Lmax, thr)
    L = sg.medfilt(Lnccf,7)
    return A, G, L, Nram

