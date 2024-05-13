import numpy as np
import numpy.matlib
import itertools
import math

def SLD(N, m):
    """Generates approximately N uniformly distributed points using the Simplex-Lattice Design with two layers"""
    H1 = 1
    while (math.comb(H1+m, m-1) <= N):
        H1 += 1
    W = nchoosek(list(range(1, H1+m)), m-1)-np.matlib.repmat(np.arange(0, m-1), math.comb(H1+m-1, m-1), 1)-1
    W = (np.hstack((W, np.zeros([len(W), 1])+H1))-np.hstack((np.zeros([len(W), 1]), W)))/H1
    if (H1 < m):
        H2 = 0
        while (math.comb(H1+m-1, m-1)+math.comb(H2+m, m-1) <= N):
            H2 += 1
        if (H2 > 0):
            W2 = nchoosek(list(range(1, H2+m)), m-1)-np.matlib.repmat(np.arange(0, m-1), math.comb(H2+m-1, m-1), 1)-1
            W2 = (np.hstack((W2, np.zeros([len(W2), 1])+H2))-np.hstack((np.zeros([len(W2), 1]), W2)))/H2
            W2 = W2/2+1/(2*m)
            W = np.vstack((W, W2))
    N = len(W)
    np.savetxt('SLD'+'_{0:0=2d}D'.format(m)+'_'+str(N)+'.pof', W, '%.6f', header=str(N)+' '+str(m))
    return

def nchoosek(v, k):
    """Returns a matrix with all combinations of v taken k at a time"""
    return np.array(list(itertools.combinations(v, k)))
