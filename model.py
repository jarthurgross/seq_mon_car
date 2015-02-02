"""Chris' model code from the IPython notebook

"""

from __future__ import division, print_function
import numpy as np
from scipy.stats import binom
from qinfer.abstract_model import Model


class HaarTestModel(Model):
    """
    Represents an experimental system with unknown quantum state.
    
    Measurements are assumed to be tests of Haar random states.
    """    
    def __init__(self, n_qubits = 1):
        
        self.dim = int(2**n_qubits)
        
        super(HaarTestModel, self).__init__()
    
    @property
    def n_modelparams(self):
        return 2*self.dim-2
        
    @property
    def expparams_dtype(self):
        return [('state', 'cfloat', (self.dim,)), ('n_meas', 'int')]

    def are_models_valid(self,modelparams):
        return np.sum(modelparams**2,1)<=1
            
    def n_outcomes(self, expparams):
        return expparams['n_meas']+1
        
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Calculates the likelihood function at the states specified 
        by modelparams and measurement specified by expparams.
        This is given by the Born rule and is the probability of
        outcomes given the state and measurement state.
        
        Parameters
        ----------
        outcomes = 
            measurement outcome, the number of successful tests
        expparams = 
            quantum state specified as complex vector
        modelparams = 
            quantum state specified as a real vector (Re[psi], Im[psi])
        """
        
        # How many states are we going to have to vectorize over
        n_particles = modelparams.shape[0]
        psi = 1j* np.zeros([n_particles,self.dim])
        
        # the first dim numbers will be the real parts
        psi[:,:-1] = modelparams[:,:self.dim-1] + 1j* modelparams[:,self.dim-1:]
        
        psi[:,-1] = np.sqrt(1 - np.sum(psi.conj()*psi,1))
        
        # Now we look at the inner product to get the bare probably
        phi = expparams['state']
        
        pr_success = (np.abs(np.sum(psi.conj()*phi,1))**2)[:,np.newaxis]
        
        # Now calculate the binomial success probability
        L = np.zeros((outcomes.shape[0],modelparams.shape[0],expparams.shape[0]))
        
        for idx_o in xrange(outcomes.shape[0]):     
            dist = binom(expparams['n_meas'],pr_success)
            L[idx_o,:] = dist.pmf(outcomes[idx_o])            
        
        return L
    
    def param2vec(self,params):
        dim = params.shape[1]/2+1
        psi = 1j*np.zeros([1,dim])
        psi[:,:dim-1] = params[:,:dim-1]+1j*params[:,dim-1:]
        psi[:,-1] = np.sqrt(1 - np.abs(np.sum(psi.conj()*psi,1)))
        return psi
        
    def fidelity(self,psi,phi):
        return np.abs(np.sum(psi.conj()*phi,1))**2
