"""Chris' distribution code from the IPython notebook, some of which uses my
code in bloch_distribution and qubit_dst

"""

from __future__ import division, print_function
import numpy as np
import scipy.linalg as la
from bloch_distribution.sampling import get_state_samples
from .model import HaarTestModel

class HaarDistribution(object):
    """This object doesn't return elements of C^2. To get state vectors, use
    StateHaarDistribution.

    """
    def __init__(self, n_qubits=1):
        self.dim = int(2**n_qubits)
    
    @property
    def n_rvs(self):
        return 2*self.dim-2

    def sample(self, n=1):
        
        # see e.g. http://arxiv.org/abs/math-ph/0609050v2
        
        samples = np.zeros([n,2*self.dim-2])
        
        for idx in xrange(n):
            z = (np.random.randn(self.dim,self.dim) + 1j*np.random.randn(self.dim,self.dim))/np.sqrt(2)
            q,r = la.qr(z)
            d = np.diag(r)
    
            ph = d/np.abs(d)
            ph = np.diag(ph)
    
            # canonical state
            psi0 = np.zeros(self.dim)
            psi0[0] = 1
            foo = np.dot(np.dot(q,ph),psi0)
            
            # we are going to chop one of the entries but let's mod out the phase first
            foo = foo * np.exp(-1j* np.arctan2((foo[-1]).real,(foo[-1]).imag))
            
            samples[idx,:] = np.concatenate(((foo[:-1]).real,(foo[:-1]).imag))
        
        return samples

class StateHaarDistribution(HaarDistribution):
    """This class should return vectors in C^2 as samples.

    """

    def __init__(self, n_qubits=1):
        self.model = HaarTestModel(n_qubits=n_qubits)
        super(StateHaarDistribution, self).__init__(n_qubits=n_qubits)

    def sample(self, n=1):
        samples = [super(StateHaarDistribution, self).sample() for m in
                   xrange(n)]
        return np.array([self.model.param2vec(sample)[0] for sample in
                         samples]).T

class MUBDistribution(object):
    def __init__(self):
            self.vecs = np.array([[np.sqrt(2),0],[1,1],[1,1j]])/np.sqrt(2)
        
    @property
    def n_rvs(self):
        pass

    def sample(self, n=1):
        
        samples = 1j*np.zeros([n,2])
        
        for idx in xrange(n):
            idr = np.random.randint(0,3)
            samples[idx,:] = self.vecs[idr,:]
        
        return samples

class TransposedMUBDistribution(MUBDistribution):
    def sample(self, n=1):
        return super(TransposedMUBDistribution, self).sample(n).T

class WeakMeasDistribution(object):
    def __init__(self, bounds = [-8,8], eps = 0.05, res = 100):
        self.bounds = bounds
        self.eps = eps
        self.res = res
        
    @property
    def n_rvs(self):
        pass

    def sample(self, n = 1):
        
        samples = get_state_samples(self.bounds[0],self.bounds[1],self.res,self.eps,n);
               
        return samples
