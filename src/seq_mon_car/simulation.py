"""Chris' function for running the SMC simulations

"""

from __future__ import division, print_function
import numpy as np
from model import HaarTestModel
from dists import HaarDistribution, MUBDistribution, WeakMeasDistribution
from qubit_dst.dst_povm_sampling import DSTDistribution
from qinfer.smc import SMCUpdater
from qinfer.resamplers import LiuWestResampler
import scipy.linalg as la
import time


def fid_smc(n_meas, K, n_qubits=1, n_trials=100, n_particles=1000, n_rec=15):
    """
    Evaluates the average fidelity incurred by using sequential Monte Carlo
    (SMC) to estimate pure states.

    :param n_meas:      The number of copies of the system given in each
                        tomographic run
    :type n_meas:       Integer
    :param K:           Number of single-shot measurements
    :type K:            Integer
    :param n_trials:    Number of times to run the SMC estimation procedure.
    :type n_trials:     Integer
    :param n_particles: Number of SMC particles to use.
    :type n_particles:  Integer
    :param n_rec:       Number of place to record data (on a log scale)
    :type n_rec:        Integer
    :returns:           Dictionary of various fidelities and timings
    :return type:       Dictionary

    """
    dim = int(2**n_qubits)
    # Record data on a logarithmic scale
    rec_idx = np.unique(np.round(np.logspace(0,np.log10(n_meas),n_rec)))
    n_rec = rec_idx.shape[0]

    # Allocate arrays to hold results.
    fidelity_mub = np.empty((n_trials,n_rec))
    fidelity_opt = np.empty((n_trials,n_rec))
    fidelity_WM  = np.empty((n_trials,n_rec))
    fidelity_DST = np.empty((n_trials,n_rec))

    # Instantiate models and distributions
    model = HaarTestModel(n_qubits = n_qubits)
    prior = HaarDistribution(n_qubits = n_qubits)
    measMUB = MUBDistribution()
    measWM = WeakMeasDistribution(eps = 0.05)
    measDST = DSTDistribution(0.1)

    timing   = np.empty((n_trials,))

    # Make and show a progress bar.
    '''
    prog = ProgressBar()
    prog.show()
    '''

    for idx_trial in xrange(n_trials):

        # Pick a random true state and instantiate the Bayes updater
        true_state = prior.sample()
        true_vec   = model.param2vec(true_state)

        updater_opt = SMCUpdater(model, n_particles, prior, resampler=LiuWestResampler(a=0.95, h = None))
        updater_mub = SMCUpdater(model, n_particles, prior, resampler=LiuWestResampler(a=0.95, h = None))
        updater_WM  = SMCUpdater(model, n_particles, prior, resampler=LiuWestResampler(a=0.95, h = None))
        updater_DST = SMCUpdater(model, n_particles, prior, resampler=LiuWestResampler(a=0.95, h = None))


        # Record the start time.
        tic = time.time()

        idx_rec = 0 
        for idx_meas in xrange(n_meas):
            # Choose a random measurement direction
            foo = prior.sample()
            meas_opt = model.param2vec(foo)
            meas_mub = measMUB.sample()
            meas_WM  = measWM.sample()[:,0]
            meas_DST = measDST.sample()[:,0]

            expparams_opt = np.array([(meas_opt,K)], dtype=model.expparams_dtype)
            expparams_mub = np.array([(meas_mub,K)], dtype=model.expparams_dtype)
            expparams_WM  = np.array([(meas_WM,K)], dtype=model.expparams_dtype)
            expparams_DST = np.array([(meas_DST,K)], dtype=model.expparams_dtype)

            # Simulate data and update
            data_opt = model.simulate_experiment(true_state, expparams_opt)
            updater_opt.update(data_opt, expparams_opt)

            data_mub = model.simulate_experiment(true_state, expparams_mub)
            updater_mub.update(data_mub, expparams_mub)

            data_WM = model.simulate_experiment(true_state, expparams_WM)
            updater_WM.update(data_WM, expparams_WM)

            data_DST = model.simulate_experiment(true_state, expparams_DST)
            updater_DST.update(data_DST, expparams_DST)

            if idx_meas+1 in rec_idx:
                # Generate the estimated state -> average then maximal eigenvector

                weights = updater_opt.particle_weights
                locs    = updater_opt.particle_locations

                ave_state = 1j*np.zeros([dim,dim])

                for idx_locs in xrange(n_particles):
                    psi = model.param2vec(locs[idx_locs][np.newaxis])
                    ave_state += weights[idx_locs]*np.outer(psi,psi.conj())

                eigs = la.eig(ave_state)
                max_eig = eigs[1][:,np.argmax(eigs[0])]

                fidelity_opt[idx_trial,idx_rec]   = model.fidelity(true_vec,max_eig)

                #MUB
                weights = updater_mub.particle_weights
                locs    = updater_mub.particle_locations

                ave_state = 1j*np.zeros([dim,dim])

                for idx_locs in xrange(n_particles):
                    psi = model.param2vec(locs[idx_locs][np.newaxis])
                    ave_state += weights[idx_locs]*np.outer(psi,psi.conj())

                eigs = la.eig(ave_state)
                max_eig = eigs[1][:,np.argmax(eigs[0])]

                fidelity_mub[idx_trial,idx_rec]   = model.fidelity(true_vec,max_eig)

                #Weak Measurement
                weights = updater_WM.particle_weights
                locs    = updater_WM.particle_locations

                ave_state = 1j*np.zeros([dim,dim])

                for idx_locs in xrange(n_particles):
                    psi = model.param2vec(locs[idx_locs][np.newaxis])
                    ave_state += weights[idx_locs]*np.outer(psi,psi.conj())

                eigs = la.eig(ave_state)
                max_eig = eigs[1][:,np.argmax(eigs[0])]

                fidelity_WM[idx_trial,idx_rec]   = model.fidelity(true_vec,max_eig)

                #DST Measurement
                weights = updater_DST.particle_weights
                locs    = updater_DST.particle_locations

                ave_state = 1j*np.zeros([dim,dim])

                for idx_locs in xrange(n_particles):
                    psi = model.param2vec(locs[idx_locs][np.newaxis])
                    ave_state += weights[idx_locs]*np.outer(psi,psi.conj())

                eigs = la.eig(ave_state)
                max_eig = eigs[1][:,np.argmax(eigs[0])]

                fidelity_DST[idx_trial,idx_rec]   = model.fidelity(true_vec,max_eig)

                idx_rec += 1

        # Record how long it took us.
        timing[idx_trial] = time.time() - tic

        print(100 * ((idx_trial + 1) / n_trials))
        # prog.value = 100 * ((idx_trial + 1) / n_trials)

    return {
        'fidelity_opt': fidelity_opt,
        'fidelity_mub': fidelity_mub,
        'fidelity_WM': fidelity_WM,
        'fidelity_DST': fidelity_DST,
        'timing'  : timing
    }

def sim_qubit_fid(n_meas, n_meas_rep, meas_dist, n_trials=100, n_particles=1000,
                  n_rec=15):
    r"""Calculates the average fidelity of the optimal estimator (approximated
    by SMC) averaged over Haar random pure states and a random sample of
    measurement outcomes. The estimator is calculated at a given number of
    interim times throughout the tomographic process.

    :param n_meas:      The number of copies of the system given in each
                        tomographic run
    :type n_meas:       Integer
    :param n_meas_rep:  The number of measurement outcomes to average the
                        fidelity over for each copy of the system in a
                        tomographic run
    :type n_meas_rep:   Integer
    :param meas_dist:   Object defining the distribution from which to draw
                        measurement directions
    :type meas_dist:    Object possessing `sample(n)` function that returns a
                        numpy.array((2, n)) of unit vectors in
                        :math:`\mathbb{C}^2`
    :param n_trials:    The number of tomographic runs (aka samples from the
                        pure state prior) the fidelity is averaged over
    :type n_trials:     Integer
    :param n_particles: Number of SMC particles to use
    :type n_particles:  Integer
    :param n_rec:       Number of places to record average fidelity (on a log
                        scale)
    :type n_rec:        Integer
    :returns:           An array with the calculated average fidelities at the
                        specified times for all tomographic runs
    :return type:       numpy.array((n_trials, n_rec))

    """
    n_qubits = 1    # This function isn't guaranteed to generalize by changing
                    # this value, but it is included here to aid readability and
                    # aid any future generalization efforts
    dim = 2*n_qubits
    # Record data on a logarithmic scale
    rec_idxs = np.unique(np.round(np.logspace(0, np.log10(n_meas), n_rec)))
    n_rec = rec_idxs.shape[0]

    # Allocate result array
    fidelities = np.empty((n_trials, n_rec))

    # Instantiate model and state prior
    model = HaarTestModel(n_qubits=n_qubits)
    prior = HaarDistribution(n_qubits=n_qubits)

    # Sample all the measurement directions used at once (since some samplers
    # might be more efficient doing things this way)
    raw_meas_dirs = meas_dist.sample(n_trials*n_meas)
    # Reshape the measurement directions to be a n_trials x n_meas array of unit
    # vectors in C^2
    meas_dirs = np.reshape(raw_meas_dirs.T, (n_trials, n_meas, 2))

    for trial_idx in xrange(n_trials):

        # Pick a random true state and instantiate the Bayes updater
        true_state = prior.sample()
        true_vec = model.param2vec(true_state)

        updater = SMCUpdater(model, n_particles, prior,
                             resampler=LiuWestResampler(a=0.95, h=None))

        rec_idx = 0
        for meas_idx in xrange(n_meas):
            meas_dir = meas_dirs[trial_idx, meas_idx]

            # Set the experimental parameters for the measurement
            expparams = np.array([(meas_dir, n_meas_rep)],
                                 dtype=model.expparams_dtype)

            # Simulate data and update
            data = model.simulate_experiment(true_state, expparams)
            updater.update(data, expparams)

            if meas_idx + 1 in rec_idxs:
                # Generate the estimated state -> average then maximal
                # eigenvector

                weights = updater.particle_weights
                locs = updater.particle_locations

                avg_state = 1j*np.zeros([dim, dim])

                for idx_locs in xrange(n_particles):
                    psi = model.param2vec(locs[idx_locs][np.newaxis])
                    avg_state += weights[idx_locs]*np.outer(psi, psi.conj())

                eigs = la.eig(avg_state)
                max_eig = eigs[1][:, np.argmax(eigs[0])]

                fidelities[trial_idx, rec_idx] = model.fidelity(true_vec,
                                                                max_eig)

                rec_idx += 1

        # Give progress updates
        print(100 * ((trial_idx + 1) / n_trials))

    return fidelities
