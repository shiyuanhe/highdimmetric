import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
import ot
from progressbar import progressbar as pbar


def wass1dim(data1, data2, numBins = 200):
    ''' Compare two one-dimensional arrays by the 
    Wasserstein metric (https://en.wikipedia.org/wiki/Wasserstein_metric).
    The input data should have outliers removed.
    
    Parameters
    ----------
        data1, data2: two one-dimensional arrays to compare.
        numBins: the number of bins.
        
    Outputs
    -------
        result: the computed Wasserstein metric.
        
    '''
    numBins = 200 ## number of bins
    upper = np.max( (data1.max(), data2.max() ) )
    lower = np.min( (data1.min(), data2.min() ) )
    xbins = np.linspace(lower, upper, numBins + 1)
    density1, _ = np.histogram(data1, density = False, bins = xbins)
    density2, _ = np.histogram(data2, density = False, bins = xbins)
    density1 = density1 / np.sum(density1)
    density2 = density2 / np.sum(density2)
    
    # pairwise distance matrix between bins
    distMat = distance_matrix(xbins[1:].reshape(numBins,1), 
                              xbins[1:].reshape(numBins,1))
    M = distMat
    T = ot.emd(density1, density2, M) # optimal transport matrix
    result = np.sum(T*M) # the objective data
    return result


def compareDensity(data1, data2):
    ''' Compare two multi-dimensional arrays by the 
    Wasserstein metric (https://en.wikipedia.org/wiki/Wasserstein_metric).
    The input data should have outliers removed before applying this funciton.
    The multidmensional input data is projected onto mutiple directions. 
    The Wasserstein metric is computed on each projected result. 
    This function returns the averaged metrics and its standard error. 
    
    
    Parameters
    ----------
        data1: the first multi-dimensional dataset. Each row is 
                an observation. Each column is a covariate. 
        data2: the second multi-dimensional dataset.
        numBins: the number of bins.
        K: the number of trial random projections.
        
    Outputs
    -------
        mu, sigma: the average discrepency measure and its standard error.
        
    '''
    K = 2000
    result = np.zeros(K)
    pCovariate = data1.shape[1]
    for i in pbar(range(K)):
        # random projection onto one dimension
        transMat = np.random.normal(size = (pCovariate, 1))
        transMat = transMat / np.linalg.norm(transMat, 'fro')
        data1_proj = data1 @ transMat
        data2_proj = data2 @ transMat
        # record the discrepency on the projected dimension
        # between two datasets.
        result[i] = wass1dim(data1_proj, data2_proj)
    return result.mean(), result.std()/np.sqrt(K)





if __name__=="__main__":
    # test on two dataset of 1000-by-5
    nObs = 1000 # number of observations
    pCov = 5 # number of variables
    
    # The first data is generated from Gaussian
    testData1 = np.random.normal(size = (nObs, pCov))
    # The second data is generated from Exponential
    testData2 = np.random.exponential(size = (nObs, pCov))
    print(compareDensity(testData1, testData2))

    