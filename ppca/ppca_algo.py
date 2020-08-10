import numpy as np
from numpy.linalg import inv,svd


class ML_ppca(object):

    def __init__(self,X,dim_q=2,sigma=1.0):
        """Initialization of the class ML_ppca for the Maximum Likelihood method
        Parameter
        ---------
        X: array-like (N,D)
            data points
        sigma: int
            covariance
        dim_q: int
            dimension of the latent space
        """

        self.X = X
        self.dim_q =dim_q
        self.prev_sigma = sigma


    def fit(self):
        """
        Method for fitting using the ML.
        Compute the mean of the obsvervable, the eigen values and the eigen vectors from the SVD decomposition,
        the weighting matrix W(D,q) that represent the mapping of the lating space to that of the principal subspace,
        the variance lost sigma in the projection over the dimension decreased.

        """

        D = self.X.shape[1]
        self.mu = np.mean(self.X,axis=0)[:,np.newaxis]
        [U,S,V] =  svd(self.X.T - self.mu)
        eig_val = S[:self.dim_q]**2
        self.W = U[:,:self.dim_q].dot(np.sqrt(np.diag(np.maximum(0,eig_val -self.prev_sigma))))
        self.sigma = 1.0/(D - self.dim_q) * np.sum(eig_val[self.dim_q:])



    def transfom(self):
        """
        Method for getting the latent space of the data points.
        Compute
        Return
        ------
        z: array-like (q,N)
            data points in the latent space
        """
        m = self.W.T.dot(self.W) + self.sigma * np.eye(self.W.shape[1])
        self.z = inv(m).dot(self.W.T).dot(self.X.T - self.mu)

        return self.z


    def inf_transform(self):
        """
        Method for getting the inference data points from the latent space to the original space
        Return
        ------
        x: array-like (D,N)
            inferred data points in the original space
        """

        x = self.W.dot(self.z) + self.mu

        return x



class EM_ppca(object):
    def __init__(self,q=2,sigma= 1.0):
        self.q = q
        self.sigma = sigma




class BV_ppca(object):
    def __init__(self,q=2,sigma=1.0):
        self.q = q
        self.sigma = sigma



class MV_ppca(object):
    def __init__(self,q=2,sigma=1.0):
        self.q =q
        self.sigma = sigma


