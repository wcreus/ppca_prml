import numpy as np
from numpy.linalg import inv,svd


class ML_ppca(object):
    def __init__(self,X,dim_q=2,sigma=1.0):
        self.dim_q =dim_q
        self.prev_sigma = sigma
        self.X = X

    def fit(self):
        D = self.X.shape[1]
        self.mu = np.mean(self.X,axis=0)[:,np.newaxis]
        [U,S,V] =  svd(self.X.T - self.mu)
        eig_val = S[:self.dim_q]**2
        self.W = U[:,:self.dim_q].dot(np.sqrt(np.diag(np.maximum(0,eig_val -self.prev_sigma))))
        self.sigma = 1.0/(D - self.dim_q) * np.sum(eig_val[self.dim_q:])




    def transfom(self):

        m = self.W.T.dot(self.W) + self.sigma
        self.z = inv(m).dot(self.W.T).dot(self.X.T - self.mu)

        return self.z


    def inf_transform(self):

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


