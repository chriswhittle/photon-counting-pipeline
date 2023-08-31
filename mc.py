import numpy as np
from inspect import signature
import emcee
import waveform

DEFAULT_WALKERS = 100
DEFAULT_STEPS = 100

def strain_mc(f, strain, noise_spectrum, p0, waveform_fn=waveform.waveform,
              priors=[(0, 1), (1, 2e3), (0, 50), (0, 2*np.pi)],
              nwalkers=DEFAULT_WALKERS, nsteps=DEFAULT_STEPS):
    """
    Perform MCMC to infer the parameters of the given waveforms.
        f: array of frequencies
        strain: (no. of frequencies) x (no. of realizations) array of strains
        noise_spectrum: noise power spectral density
        p0: initial guesses for the walkers, use e.g. actual parameters used to
            generate the strains
        waveform_fn: function that takes parameters and returns a waveform
        priors: list of tuples of the form (min, max) for each parameter
                (for uniform priors)
        nwalkers: number of walkers to use in MCMC
        nsteps: number of steps to take in MCMC
    """

    # number of parameters to infer (minus one for frequency parameter)
    ndim = len(signature(waveform_fn).parameters) - 1

    # uniform priors #TODO: update
    def log_prior(theta):
        if all([p[0] < x < p[1] for p, x in zip(priors, theta)]):
            return 0.0
        return -np.inf

    # log likelihood using Gaussian noise, e.g. (5) from PASA vol. 36 e10
    def log_likelihood(theta, f , strain, noise, waveform_fn):
        model = waveform_fn(f, *theta)
        return -0.5 * np.sum(np.abs(strain - model) ** 2 / noise + np.log(noise))
    
    # probability = prior * likelihood
    def log_probability(theta, f, strain, noise, waveform_fn):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, f, strain, noise, waveform_fn)

    for wv_ind in range(strain.shape[1]):
        wv = strain[:, wv_ind]

        pos = (p0[:, wv_ind]) * (1 + np.random.randn(nwalkers, ndim)/4)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(f, wv, noise_spectrum, waveform_fn)
        )
        sampler.run_mcmc(pos, nsteps, progress=True)
        return sampler



def count_mc(f, count, noise_spectrum, waveform_fn, nwalkers=DEFAULT_WALKERS):
    """
    Perform MCMC to infer the parameters of the given waveforms, after they have
    been filtered by output template filters and converted to photon counts.
        f: array of frequencies
        count: (no. of realizations) x (no. of templates) array of photon counts
        noise_spectrum: noise power spectral density
        waveform_fn: function that takes parameters and returns a waveform
        nwalkers: number of walkers to use in MCMC
    """
    pass