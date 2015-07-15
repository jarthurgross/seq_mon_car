"""A routine for generating measurement records that can be fed to different
estimators for comparison purposes

"""

from __future__ import division, print_function
import numpy as np
from .model import HaarTestModel
from qubit_dst.dst_povm_sampling import DSTDistribution
from qinfer.smc import SMCUpdater
from qinfer.resamplers import LiuWestResampler

#TODO: Finish implementation
def generate_records(n_meas, n_meas_rep, estimators, n_trials=100,
                     n_particles=1000):
    """Generate measurement records. (Not implemented yet)

    :param n_meas:      Number of copies of the system to be measured in each
                        trial.
    :type n_meas:       int
    :param n_meas_rep:  Number of measurement outcomes samples for each copy of
                        the system for each trial.
    :type n_meas_rep:   int
    :param estimators:  List of functions that take measurement records and
                        calculate estimates of the state that produced those
                        records.
    :type estimators:   [function, ...]
    :param n_trials:    Number of input pure states sampled from the prior
                        distribution.
    :type n_trials:     int
    :param n_particles: Number of SMC particles to use.
    :type n_particles:  int
    :returns:           An array with the calculated average fidelities at the
                        specified times for all tomographic runs
    :return type:       numpy.array((n_trials, n_rec))

    """

    n_qubits = 1    # Not tested for n_qubits > 1
    dim = int(2**n_qubits)

    # Instantiate model and state prior
    model = HaarTestModel(n_qubits=n_qubits)
    prior = HaarDistribution(n_qubits=n_qubits)

    # Sample all the measurement directions used at once (since some samplers
    # might be more efficient doing things this way)
    raw_meas_dirs = meas_dist.sample(n_trials*n_meas)
    # Reshape the measurement directions to be a n_trials x n_meas array of unit
    # vectors in C^2
    meas_dirs = np.reshape(raw_meas_dirs.T, (n_trials, n_meas, 2))
