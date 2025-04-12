import numpy as np

def frame(x, ls=160, rs=0):
    fs = ls - rs
    Nram = int(1 + np.floor((len(x) - ls) / fs))
    xram = np.array([x[i*fs:i*fs + ls] for i in range(Nram)])
    return xram

