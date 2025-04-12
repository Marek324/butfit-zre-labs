import numpy as np
import scipy.signal as sg
import librosa
import matplotlib.pyplot as plt
    

def vq_code(data, codebook):
    """
    Syntax: mind_idx, gd = vq_code(data, codebook)
    
    Coding using VQ codebook.
    M (dimensions N x P) is the matrix to code. Each column contains one vector.
    CB (dimensions L x P) is the codebook. Each column contains one code-vector.
    mind_idx (dimensions N) is the string of symbols in the range 1 to L.
    gd is the global distance normalized by N. 
    
    All distances are quadratic cepstral (x-y)'*(x-y)
    """

    N, P = data.shape
    L, _ = codebook.shape
    # now compute the distance of all vectors to all codevectors.

    D = np.zeros((N, L))
    gd=0
    for i, d in enumerate(data):
        for j, cb in enumerate(codebook):
            D[i,j] = (d-cb).T @ (d-cb)

    # now search the minima
    # D is now matrix with distances of i-th vector from j-th codebook
    mind = np.min(D, axis=1)
    mind_idx = np.argmin(D, axis=1)
    # compute the glob dist
    gd = sum (mind) / N;
    return mind_idx, gd
    
def vq_clust(M, sym, L):
    """
    Syntax:   cb, nbs = vq_clust(M, sym, L)
    
    Making VQ centroids, which will serve as new code-vectors.
    
    M (dimensions P x N) is the matrix of training vectors. 
                         Each column contains one vector.
    sym (dimensions 1 x N) is the associated string of symbols (range 1 to L).
    L  is the size of codebook. 
    CB (dimensions P x L) is the resulting codebook. 
                          Each column contains one code-vector.
    nbs (dimensions 1 x L) is the vector of numbers of training vectors 
                           associated to clusters.
    The centroids are computes using simple means.
    """
    N, P = M.shape;

    cb = np.zeros((L, P))
    nbs = np.zeros(L)
    for ii in range(L):
        indices = (sym==ii)
        nbs[ii] = np.sum(indices)
        chosen_vecs = M[indices]
        centroid = np.mean(chosen_vecs, axis=0)
        cb[ii] = centroid;
    return cb, nbs
    
def vq_split(codebook, const=0.01):
    """
     Syntax:  split_codebook = vq_split(codebook):
    
     Splitting VQ codebook. 
    
     CB (dimensions L x P) codebook. Each column contains one code-vector.
     SCB (dimensions 2L x P) is the split codebook. 
    
     Each codevector is split into two, moved in opposite directions by
     +const * c, -const * c, where c is the code-vector
    """
    
    L, P = codebook.shape
    split_codebook = np.zeros ((2*L, P))
    for ii in range(L):
      ii1=2*ii-1; 
      ii2=2*ii;
      
      c = codebook[ii]
      split_codebook[ii1] = c + const*c;
      split_codebook[ii2] = c - const*c
    return split_codebook

def lbg(data, final_size, error=0.005, init_codebook=None):
    its = int(np.ceil(np.log2(final_size)))
    true_final_size = 2**its
    # LBG
    nb_data, nb_fea = data.shape
    if init_codebook is None:
        it_from = 0
        codebook = np.zeros((1, nb_fea)) # jeden codevector, init na same 0.
        idx_s, gd = vq_code (data, codebook)

        # --- compute init cluster ---
        L=1;
        codebook, nbs = vq_clust(data, idx_s, L)
    else:
        idx_s, gd = vq_code (data, init_codebook)
        # --- compute init cluster ---
        L = init_codebook.shape[0]
        codebook, nbs = vq_clust(data, idx_s, L)
        it_from = int(np.log2(L))
    print(f'Codebook len: {L}, initial global distance: {gd:.2f}')
        
    # --- split and iterate ---
    for ii in range(it_from, its):
        codebook = vq_split(codebook)
        L = 2 * L;

        # - first code here -
        idx_s, gd = vq_code (data, codebook)
        print(f'Codebook len: {L}, initial global distance: {gd:.2f}')
        oldgd=9999999
        jj=0

        while ((oldgd - gd)/gd > error):
            # - make new code-vectors
            codebook, nbs = vq_clust (data, idx_s, L)
            
            # - and code - 
            oldgd = gd
            idx_s, gd = vq_code(data, codebook)
            print(f'Iter: {jj}: GD: {gd}', end='\r')
            jj=jj+1;
    print(f'Codebook len: {L}, final global distance: {gd:.2f}')    
    return codebook
    
def show(A, codebook, sym, i0=1, i1=2, i2=3, pca=False, title="Vector quantization", true_codebook=None):
    """
    function show (A,CB,sym)
    
    shows the clusters and centroids of VQ 
    A - data 
    codebook - codebook 
    sym - symbols produced by vq_code. 
    """ 
    to_show = A
    c_to_show = codebook
    t_c_to_show = true_codebook
    if pca:
        # normalize
        # https://towardsdatascience.com/pca-with-numpy-58917c1d0391
        mu, std = A.mean(axis=0), A.std(axis=0)
        An = (A - mu)/std
        cov = np.cov(An.T)
        e_val, e_vec = np.linalg.eig(cov)
        v_e = []
        for i in e_val:
            v_e.append((i/sum(e_val))*100)
        v_e = np.hstack(v_e)
        ind = np.sort(np.argpartition(e_val, -3)[-3:])
        projection_matrix = (e_vec.T[:][ind]).T
        to_show = A.dot(projection_matrix)
        c_to_show = codebook.dot(projection_matrix)
        if true_codebook is not None:
            t_c_to_show = t_c_to_show.dot(projection_matrix)
        i0,i1,i2 = np.arange(1,4)
        
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    colors = 'bgrcmyk'
    for ii in range(codebook.shape[0]):
      indices = (sym == ii)
      ax.scatter(to_show[indices, i0-1], to_show[indices, i1-1], to_show[indices, i2-1], marker='x', s=5, c=colors[ii % len(colors)], alpha=0.4)
      ax.scatter(c_to_show[ii, i0-1], c_to_show[ii, i1-1], c_to_show[ii, i2-1], marker='o', s=200, c=colors[ii % len(colors)])
      if true_codebook is not None:
          ax.scatter(t_c_to_show[ii, i0-1], t_c_to_show[ii, i1-1], t_c_to_show[ii, i2-1], marker='^', s=200, c='k')
      
    ax.set_xlabel(f'a{i0}')
    ax.set_ylabel(f'a{i1}')
    ax.set_zlabel(f'a{i2}')
    ax.set_title(title)
    plt.show()



