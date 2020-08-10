import numpy as np
from numpy.linalg import inv,svd
import math


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


    def infer_transform(self):
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
    """
    Initialization of the class EM_ppca for the Expectation-Maximization algorithm
    Parameter
    ---------
    X: array-like (N,D)
        data points
    sigma: int
        covariance
    dim_q: int
        dimension of the latent space
    """
    def __init__(self,X,dim_q=2,sigma= 1.0,max_iter=40):
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
        self.X =X
        self.q = dim_q
        self.sigma = sigma
        self.max_iter = max_iter

    def ell(self,w,m,mu,norm=True):
        """
        Method for computing the expected log-likelihood during the E-step
        Parameter
        ---------
        :m array-like (q,q)
            matrix related to the covariance and the mapping latent space
        :mu array-like (D,1)
            mean for each observable
        :norm bool
            Normalilsation of the LL
        Return
        ------
        ll: float
            return the value of the expected LL
        """

        ll = 0.5 * self.X.shape[0] * self.X.shape[1] * np.log(2 * math.pi*self.sigma+1e-5)


        for i in range(self.X.shape[0]):
            x_scl_i = self.X[i][:,np.newaxis] - mu
            zi = inv(m).dot(w.T).dot(x_scl_i)

            ll += 0.5 * np.trace(self.sigma*inv(m) + zi.dot(zi.T))
            ll += (2*self.sigma+1e-5)**-1 * float(x_scl_i.T.dot(x_scl_i))
            ll -= (self.sigma+1e-5)**-1 * float(zi.T.dot(w.T).dot(x_scl_i))
            ll += (2*self.sigma)**-1 * np.trace((self.sigma*inv(m)+zi.dot(zi.T)).dot(w.T).dot(w))

        ll *= -1

        if norm == True:
            ll /= float(self.X.shape[0])

        return ll



    def fit(self):
        """
        Method for computing the M-step by maximization respect with W and sigma

        """
        self.W = np.random.rand(self.X.shape[1],self.q)
        m = self.W.T.dot(self.W) + self.sigma * np.eye(self.q)
        self.mu = np.mean(self.X,axis=0)[:,np.newaxis]
        S = self.X.shape[0]**-1 * (self.X.T - self.mu).dot((self.X.T - self.mu).T)
        ll = self.ell(self.W,m,self.mu)
        #print(float(ll))
        print("Iteration 0 : LL = {:.2f}".format(ll))
        for i in range(self.max_iter):
            W_new = S.dot(self.W).dot(inv(self.sigma * np.eye(self.q) + inv(m).dot(self.W.T).dot(S).dot(self.W)))
            sigma_new = self.X.shape[1]**-1 * np.trace(S-(S).dot(self.W).dot(inv(m)).dot(W_new.T))
            m = W_new.T.dot(W_new) + sigma_new * np.eye(self.q)
            self.sigma = sigma_new
            ll = self.ell(W_new,m,self.mu)
            self.W = W_new


            print("Iteration {} : LL = {:.2f}".format(i+1,ll))

    def transform(self):
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

    def infer_transform(self):
        """
        Method for getting the inference data points from the latent space to the original space
        Return
        ------
        x: array-like (D,N)
            inferred data points in the original space
        """

        x = self.W.dot(self.z) + self.mu

        return x




class BV_ppca(object):
    def __init__(self,q=2,sigma=1.0):
        self.q = q
        self.sigma = sigma



class MV_ppca(object):
    def __init__(self,q=2,sigma=1.0):
        self.q =q
        self.sigma = sigma


