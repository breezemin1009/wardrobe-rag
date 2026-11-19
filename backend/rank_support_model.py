"""Immutable pointwise neural support with batch-invariant CPU inference."""
import numpy as np
from scipy.special import expit

class NumpyRankSupport:
    def __init__(self,mean,scale,layers):
        self.mean=np.asarray(mean,dtype=np.float64);self.scale=np.asarray(scale,dtype=np.float64)
        self.layers=[(np.asarray(w,dtype=np.float64),np.asarray(b,dtype=np.float64)) for w,b in layers]
        self.n_features_in_=len(self.mean)

    def predict_proba(self,X):
        z=(np.asarray(X,dtype=np.float64)-self.mean)/self.scale
        for i,(w,b) in enumerate(self.layers):
            z=np.concatenate([(np.repeat(v[None,:],4,axis=0)@w.T)[:1] for v in z],axis=0)+b
            if i+1<len(self.layers):z=np.tanh(z)
        p=expit(z[:,0]);return np.column_stack([1-p,p])
