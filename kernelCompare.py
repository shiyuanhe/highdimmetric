import matplotlib.pyplot as plt
import numpy as np
from progressbar import progressbar as pbar
from numba import jit
import matplotlib.pyplot as plt
import matplotlib as mpl


class kernelCompare:
    def __init__(self,D1, D2):
        self._D1 = D1
        self._D2 = D2
        self._XY = np.vstack((D1, D2))
        self._scale = self._computeScale(self._XY)
        self._n1 = len(D1)
        self._n2 = len(D2)
        

    
    def _computeScale(self,XY):
        '''Compute and determine the kernel parameter by
        mean absolute deviation
        '''
        Z = XY -  np.mean(XY,0)
        Z = np.abs(Z)
        scaleXY = np.median(Z, 0)
        return scaleXY

    def _rbf(self,z1, z2):
        diff = z1 - z2
        diff /= self._scale
        diffSq = np.sum(diff * diff,1)
        res = np.exp(-diffSq)
        return res
    
    @staticmethod
    @jit(nopython=True)
    def _MMD2ufast( X, Y, scale):
        '''Compute the unbiased MMD2u statistics in the paper. 
        $$Ek(x,x') + Ek(y,y') - 2Ek(x,y)$$
        This function implemnts a fast version in linear time. 
        '''
        n1 = len(X)
        n2 = len(Y)
        k1 = 0.0
        for i in range(n1-1):
            diff = (X[i,:] - X[i+1,:])/scale
            diffSq = np.sum(diff * diff)
            k1 += np.exp(-diffSq)
        k1 /= n1 - 1
        
        k2 = 0.0
        for i in range(n2-1):
            diff = (Y[i,:] - Y[i+1,:])/scale
            diffSq = np.sum(diff * diff)
            k2 += np.exp(-diffSq)
        k2 /= n2 - 1
        
        k3 = 0.0
        p = min(n1, n2)
        for i in range(p):
            diff = (X[i,:] - Y[i,:])/scale
            diffSq = np.sum(diff * diff)
            k3 += np.exp(-diffSq)
        k3 /= p
        result = k1 + k2 - 2*k3
        return result

    def _compute_null_dist(self,iterations=500):
        '''Compute the bootstrap null-distribution of MMD2u.
        '''
        mmd2u_null = np.zeros(iterations)
        for i in pbar(range(iterations)):
            idx = np.random.permutation(self._n1 + self._n2)
            XY_i = self._XY[idx, :]
            mmd2u_null[i] = self._MMD2ufast(XY_i[:self._n1,:], XY_i[self._n1:,], self._scale)

        return mmd2u_null

    def compute(self,iterations=500):
        '''Compute MMD^2_u, its null distribution and the p-value of the
        kernel two-sample test.
        '''
        mmd2u = self._MMD2ufast(self._D1, self._D2, self._scale)
        mmd2u_null = self._compute_null_dist(iterations)
        p_value = max(1.0/iterations,
                      (mmd2u_null > mmd2u).sum() /float(iterations))

        return mmd2u, p_value

    def plotDiff(self, coord1, coord2):
        v0min = np.min(self._XY[:,coord1])
        v1min = np.min(self._XY[:,coord2])
        v0max = np.max(self._XY[:,coord1])
        v1max = np.max(self._XY[:,coord2])
        nSeq = 50
        xSeq = np.linspace(v0min, v0max, nSeq)
        ySeq = np.linspace(v1min, v1max, nSeq)
        #xySeq = np.array(np.meshgrid(xSeq, ySeq)).T.reshape(-1,2)
        fGrid = np.zeros((nSeq, nSeq))
        znew = np.mean(self._XY, 0)
        for i in range(nSeq):
            for j in range(nSeq):
                znew[coord1] = xSeq[i]
                znew[coord2] = ySeq[j]
                #fGrid[i,j] = xSeq[i] *  xSeq[i]  ySeq[j] * ySeq[j]
                fpart1 = np.mean(self._rbf(znew, self._D1))
                fpart2 = np.mean(self._rbf(znew, self._D2))
                fGrid[i,j] = fpart1 - fpart2
        fig, ax = plt.subplots()
        vmax = np.max(np.abs(fGrid))
        vmax = max(vmax, 0.0005)
        cs = plt.contourf(xSeq, ySeq, fGrid.T, 
                          cmap = plt.get_cmap("RdBu"),
                         norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax))
        fig.colorbar(cs, ax=ax, shrink=0.9)
