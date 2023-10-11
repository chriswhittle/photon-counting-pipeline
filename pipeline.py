import sys
from dataclasses import dataclass
from inspect import signature, getmembers
import types
from warnings import warn

import math
import numpy as np
from scipy.constants import c, hbar
from scipy.special import erf, expi
from scipy.linalg import orth
import gwinc
import emcee
import pickle
import json
from multiprocessing import Pool
from pathlib import Path
import yaml
from tqdm import tqdm

import waveform


def frequency_bins(sampling_frequency, bin_width, f_low=None, f_high=None):
    """
    Generate one-sided list of frequencies with given.
    """
    # generate one-sided frequency bins up to Nyquist
    f = np.arange(bin_width, sampling_frequency/2, bin_width)

    # apply frequency bounds
    if f_low is not None:
        f = f[f >= f_low]
    if f_high is not None:
        f = f[f <= f_high]

    return f


class DetectionPipeline:

    ##############################################################
    ######################## SETUP ###############################
    ##############################################################

    def __init__(self, f_sample=16384, bin_width=1, f_low=None, f_high=None,
                 detector='CE1', waveform_func=waveform.lorentzian,
                 snr_cutoff=0.2, param_means=None, param_stds=None,
                 mcmc_mask=None, dist_priors=None, hd_gaussian=False,
                 template_params=None, parallel=True, lean=True, **kwargs):
        """
        Initialize detection pipeline with particular frequency bins within a 
        given range and for a given interferometer noise budget.

            f_sample: sampling frequency
            bin_width: frequency bin width
            f_low: lower frequency bound
            f_high: upper frequency bound
            detector: detector for which to fetch noise curves, either a string
                (gwinc canonical budget label, e.g. 'CE1', 'aLIGO') or a gwinc 
            budget object
            waveform_func: function used to generate post-merger waveform;
                should have signature waveform_func(f, *args) where f is
                a broadcast frequency array and args are the waveform parameters
            snr_cutoff: SNR threshold for detection of an event; post-merger
                waveforms will be drawn from a distribution that uses this as a
                minimum (should be ~0.1 times the CBC waveform SNR)
            param_means: list of means for waveform parameters
            param_stds: list of standard deviations for waveform parameters
            mcmc_mask: list of booleans indicating which parameters to infer
            dist_priors: list of tuples of (lower, upper) bounds for each
                inferred parameter for the astrophysical distribution
                in the format [mean_lims | std_lims] =
                [(mean_lower, mean_upper), ..., (std_lower, std_upper), ...]
            hd_gaussian: whether to assume Gaussian posteriors for the
                event parameters when computing the astrophysical posterior
                (only appropriate for events above threshold that do not have
                uniform posteriors)
            template_params: list of parameters for each template waveform for
                photon counting detection
            parallel: whether to use Multiprocessing parallelization
            lean: set to True to discard sample chains before saving
        """
        # seeded random number generator to use throughout
        self.rng = np.random.default_rng(0)

        # generate frequency bins
        self.bw = bin_width
        self.f = frequency_bins(f_sample, bin_width, f_low, f_high)

        # load detector budget
        if isinstance(detector, str):
            self.detector = gwinc.load_budget(detector)
        else:
            self.detector = detector

        # compute noise curves
        trace = self.detector.run(freq=self.f)

        self.noise_quantum = trace['QuantumVacuum'].psd
        self.noise_classical = trace.psd - self.noise_quantum
        self.noise_total = trace.psd
        self.log_noise_total = np.log(self.noise_total)

        # precompute calibration factor for photon probability
        self.ifo_calibration = self.compute_ifo_calibration()

        # function used to compute (unit) waveform
        if isinstance(waveform_func, str):
            self.waveform_func = waveform.WAVEFORM_FUNCTIONS[waveform_func]
        else:
            self.waveform_func = waveform_func

        # SNR cutoff for post-merger waveform; used for building event
        # distribution on which we do MCMC
        self.snr_cutoff = snr_cutoff

        # instantiate parameter means and variances and MCMC mask and priors
        param_count = len(signature(self.waveform_func).parameters) - 1
        if param_means is None:
            param_means = [1]*param_count
        if param_stds is None:
            param_stds = [0]*param_count
        if mcmc_mask is None:
            mcmc_mask = [True]*param_count

        self.param_means = np.array(param_means, dtype=np.float64)
        self.param_stds = np.array(param_stds, dtype=np.float64)
        self.mcmc_mask = mcmc_mask
        self.ndim_p = np.sum(mcmc_mask)
        self.parallel = parallel

        # priors for parameters of astrophysical distribution
        self.dist_priors = dist_priors
        assert(self.dist_priors is not None)
        assert(len(self.dist_priors) == 2*self.ndim_p)

        # scatter for MCMC walkers
        self._DIST_WALKER_STD = [
            (p[1]-p[0])*self.WALKER_STD for p in self.dist_priors
        ]

        # whether to assume Gaussian posteriors for event parameters
        self.hd_gaussian = hd_gaussian
        if self.hd_gaussian and self.snr_cutoff < 1:
            warn("Gaussian posteriors should not be assumed for subthreshold events")

        # parameters with non-zero standard deviation need to be included in
        # MCMC; parameters not inferred should have zero variance
        for i, std in enumerate(self.param_stds):
            assert(self.mcmc_mask[i] == (std > 0))

        # compute template waveforms given matrix of parameter values
        if template_params is not None:
            self.templates = self.waveform(np.array(template_params).T)

            # orthonormalize waveform templates
            self.templates = orth(self.templates)

        # precompute some other useful matrices
        self._event_prior_sign_mat = (
            (1 - 2*np.eye(2))[np.newaxis, :, :].repeat(self.ndim_p, axis=0)
        )

        # whether to produce a lean results file
        self.lean = lean

    ##############################################################
    ###################### WAVEFORM METHODS ######################
    ##############################################################

    def draw_params(self, N, means=None, stds=None):
        """
        Draw parameter values from Gaussian distributions with given means and
        standard deviations.

            N: number of samples to draw
            means: means of parameter distributions
            stds: stds of parameter distributions

        Returns a numpy array of shape (no. of parameters, N) containing the
        drawn values.
        """
        # use true distribution values for this pipeline if no parameters given
        if means is None:
            means = self.param_means
        if stds is None:
            stds = self.param_stds

        # transpose so no. of samples is last dimension for broadcasting
        return self.rng.normal(means, stds, size=(N, len(self.param_means))).T

    def inner_product(self, a, b):
        """
        Compute the inner product of two signals a and b. Equation (104)
        from McCuller22.
        """
        return self.bw * np.sum(a * np.conj(b), axis=0)

    def compute_snr(self, wv):
        """
        Compute the SNR of the given waveform:
        SNR = sqrt(2 * integral(|waveform|^2 / noise))
        """
        # duplicate noise across a new axis for broadcasting
        # if params is 2-dimensional
        wv = np.array(wv)
        if len(wv.shape) == 2:
            noise = self.noise_total[:, np.newaxis].repeat(wv.shape[1], axis=1)
        else:
            noise = self.noise_total
        return np.sqrt(2 *
                       np.real(self.inner_product(wv / noise, wv))
                       )

    def sample_events(self, N, return_params=False, means=None, stds=None):
        """
        Sample events from the astrophysical distribution of waveform parameters
        and the distribution of distances.
            N: number of events
            return_params: whether to return a tuple of
                (drawn parameters, waveforms) instead of just the waveforms
            means: means of astro distribution to use
            stds: stds of astro distribution to use

        Returns a numpy array of shape (no. of frequencies, N) containing the
        sampled waveforms unless return_params is True, in which case it returns
        a tuple of (parameters, waveforms) where parameters is a numpy array of
        shape (no. of parameters, N).
        """
        # use true distribution values for this pipeline if none given
        if means is None:
            means = self.param_means
        if stds is None:
            stds = self.param_stds

        # SNR ~ amplitude ~ distance
        # distance CDF = (distance/distance_max)^3
        # SNR CDF = 1 - (SNR_min/SNR)^3
        # SNR = SNR_min / (1 - random)*(1/3)
        snr_samples = self.snr_cutoff / (1 - self.rng.random(N))**(1/3)

        # draw parameters from Gaussian distributions
        param_samples = self.draw_params(N, means, stds)
        waveforms = self.waveform(param_samples)

        # compute SNR of waveforms to renormalize (since waveforms of unit
        # normalization will have varying SNRs)
        unit_snrs = self.compute_snr(waveforms)
        amplitudes = snr_samples / unit_snrs
        waveforms *= amplitudes

        # return waveforms and parameters if requested
        if return_params:
            combined_params = np.vstack((amplitudes, param_samples))
            return combined_params, waveforms
        else:
            return waveforms

    def waveform(self, params):
        """
        Compute and return unit wavelets with the given parameters.
            f: numpy array of equally-spaced frequency bins
            params: list/array of waveform parameter values with shape
                (parameters, samples)
        """
        # duplicate f across a new axis for broadcasting
        # if params is 2-dimensional
        params = np.array(params)
        if len(params.shape) == 2:
            f = self.f[:, np.newaxis].repeat(params.shape[1], axis=1)
        else:
            f = self.f

        # need to transpose to map 2nd dimension (parameters) to function
        # parameters
        wavelet = self.waveform_func(f, *params)

        # normalize wavelet to unit norm
        wavelet /= np.sqrt(self.inner_product(wavelet, wavelet))

        return wavelet

    ##############################################################
    ###################### DETECTOR METHODS ######################
    ##############################################################

    def simulate_noise(self, N, psd):
        """
        Generate N realizations of noise at the specified frequencies shaped by
        the given power spectral density.
            N: number of realizations to generate
            psd: noise power spectral density
        """
        # simulate white noise with random phase (normalized to have unit standard
        # deviation in magnitude)
        white_noise = (self.rng.normal(size=(len(self.f), N))
                       + 1j*self.rng.normal(size=(len(self.f), N))) / np.sqrt(2)

        # scale noise by PSD
        return white_noise * np.sqrt(psd[:, np.newaxis])

    def compute_ifo_calibration(self):
        """
        Compute the calibration factor for converting strain into the IFO
        mean output field (in units of counts).
        """
        # NB: this assumes transfer function is flat, i.e. (32) in McCuller22;
        # neglects any recycling in IFO.
        Parm = gwinc.ifo.noises.ifo_power(self.detector.ifo).parm
        Larm = self.detector.ifo.Infrastructure.Length
        wavelength = self.detector.ifo.Laser.Wavelength
        wavenumber = 2 * np.pi / wavelength
        omega = c * wavenumber

        # change in length -> quanta at output; eq. (32) in McCuller22
        g = 2 * wavenumber * np.sqrt(Parm / (2*hbar*omega))

        return g * Larm

    def no_photon_prob(self, strain):
        """
        Given a series of strain spectra, convert to output optical power and then
        subsequently to "probability of observing zero photos" in each optical mode
        at the output.
            strain: (no. of frequencies, no. of realizations) array of strain
            budget: gwinc budget object, e.g. gwinc.load_budget(ifo) where ifo is
                one of 'aLIGO', 'CE1' etc.
        Returns matrix (no. of realizations, no. of templates) of probabilities
        """
        # output fields
        output_fields = strain * self.ifo_calibration

        # mean fields in output templates
        # units of sqrt(quanta) TODO: check this
        mean_counts = np.zeros(
            (strain.shape[1], self.templates.shape[1]), dtype=complex
        )
        for i, template in enumerate(self.templates.T):
            mean_counts[:, i] = self.inner_product(
                template[:, np.newaxis].repeat(strain.shape[1], axis=1),
                output_fields
            )

        # convert mean counts to probability of zero photons
        # Pr(n=0) = |<0|alpha>|^2 = exp(-|alpha|^2)
        return np.exp(-np.abs(mean_counts)**2)

    def count_photons(self, prob):
        """
        Given a matrix of probabilities of observing zero photons in each 
        realization for each template mode, randomly populate each with 0 or 1
        photons.
            prob: (no. of realizations, no. of templates) matrix of probabilities
        Returns a matrix of the same shape with 0s and 1s.
        """
        return np.array(self.rng.random(prob.shape) > prob, dtype=int)

    ##############################################################
    ###################### MCMC METHODS ##########################
    ##############################################################

    # default parameters for MCMC
    DEFAULT_WALKERS = 50
    DEFAULT_STEPS_EVENT = 1500
    DEFAULT_STEPS_DIST = 2000

    # constants for setting up walker initial positions
    WALKER_STD = 3e-1
    WALKER_STD_MAX = 1e-1
    EPS = 1e-3

    # prior, likelihood, probability functions
    # these need to be top-level to be pickle-able by Multiprocessing

    def log_prior_event(self, theta):
        """
        Compute the log-prior for the given parameters of a single event, the
        result of integrating a Gaussian over uniform priors on the mean
        and standard deviation.
            theta: array of parameters
        """
        # amplitude should always be positive
        if theta[0] <= 0:
            return -np.inf

        # get priors for means and stds of event parameters;
        # each numpy array has dimensions (no. of params, 2)
        mu_p = np.array(self.dist_priors[:self.ndim_p])
        sigma_p = np.array(self.dist_priors[self.ndim_p:])

        # standard deviations of inferred parameters should always be positive
        if np.any(sigma_p <= 0):
            return -np.inf

        # make an effective meshgrid on the last dimension of mu_p, sigma_p
        MU_P = mu_p[:, :, np.newaxis].repeat(2, axis=2)
        SIGMA_P = sigma_p[:, np.newaxis, :].repeat(2, axis=1)

        # broadcast theta to shape (no. of params, 2, 2)
        t = theta[1:, np.newaxis, np.newaxis].repeat(2, 1).repeat(2, 2)

        prior_terms = SIGMA_P * erf((t - MU_P) / np.sqrt(2) / SIGMA_P)

        with np.errstate(invalid='ignore'):
            expi_terms = (
                (t - MU_P) * expi(-(t - MU_P)**2/2/SIGMA_P**2)
                / np.sqrt(2*np.pi)
            )

        # only add the expi term when not infinite (in which case they
        # cancel)
        inds = (t != MU_P)
        prior_terms[inds] -= expi_terms[inds]

        # multiply by appropriate sign
        prior_terms *= self._event_prior_sign_mat

        prior = prior_terms.sum(axis=(1, 2)).prod()

        # use math.log for scalar performance
        return -np.inf if prior <= 0 else math.log(prior)

    def log_prior(self, theta, priors):
        """
        Compute the uniform log-prior for the given parameters.
            theta: array of parameters
            priors: list of tuples of (lower, upper) bounds for each parameter
        """
        if all([p[0] <= x <= p[1] for p, x in zip(priors, theta)]):
            return 0.0
        return -np.inf

    def log_prior_dist(self, theta):
        """
        Compute the uniform log-prior for the given parameters of the
        astrophysical distribution.
            theta: array of parameters
        """
        return self.log_prior(theta, self.dist_priors)

    def log_likelihood_event(self, theta, strain):
        """
        Log-likelihood function using Gaussian noise, e.g. (5) from
        PASA vol. 36 e10.
            theta: array of parameters
            strain: strain waveform
        """
        # use distribution mean values for parameters that are not being
        # inferred
        params = self.param_means.copy()
        params[self.mcmc_mask] = theta[1:]

        # compute waveform and scale by amplitude
        model = self.waveform(params) * theta[0]

        # compute log likelihood for Gaussian noise
        return -0.5 * np.sum(
            np.abs(strain - model) ** 2 / self.noise_total
            + self.log_noise_total
        )

    def log_likelihood_hd(self, theta, event_post, wv_ind, nint=10000):
        """
        Log-likelihood function for hyper-parameter MCMC on homodyne detection,
        doing full integration based on event posteriors.
            theta: array of parameters
            wv_ind: index of current event (up to which we should sample
                when integrating over event posteriors)
            event_post: posterior distribution of event parameters, numpy array
                of shape (no. of events, no. of MCMC samples, no. of params)
        """
        # separate inferred parameters into means and stds
        means = theta[:self.ndim_p]
        stds = theta[self.ndim_p:]

        # randomly select events up to current index to sample
        event_inds = self.rng.choice(wv_ind+1, nint)
        # randomly select MCMC chain samples
        sample_inds = self.rng.choice(event_post.shape[1], nint)
        sampled_events = event_post[event_inds, sample_inds, 1:]

        return -0.5 * np.sum(
            (sampled_events - means)**2 / stds**2
            + np.log(stds**2)
        )

    def log_likelihood_hd_gaussian(self, theta, param_means, param_stds):
        """
        Log-likelihood function for hyper-parameter MCMC on homodyne detection,
        assuming the event posteriors are Gaussian. Inappropriate for
        subthreshold events, which will result in uninformative (uniform)
        posteriors.
            theta: array of parameters
            param_means: matrix of means of inferred parameters
            param_stds: matrix of standard deviations of inferred parameters
        """
        # separate inferred parameters into means and stds
        means = theta[:self.ndim_p]
        stds = theta[self.ndim_p:]

        # pre-compute sum of astrophysical stds and inferred event-wise stds
        std_sum2 = stds**2 + param_stds**2

        # compute log likelihood for Gaussian distributions
        return -0.5 * np.sum(
            (means - param_means)**2 / std_sum2 + np.log(std_sum2)
        )

    def log_likelihood_pc(self, theta, counts, nint):
        """
        Log-likelihood function for photon counting.
            theta: array of parameters
            counts: array of photon counts
        """
        # use distribution mean values for parameters that are not being
        # inferred
        means = self.param_means.copy()
        stds = self.param_stds.copy()

        # separate inferred parameters into means and stds
        means[self.mcmc_mask] = theta[:self.ndim_p]
        stds[self.mcmc_mask] = theta[self.ndim_p:]

        # for this astrophysical distribution, compute the probabilities
        # of observing zero photons in each template mode by sampling
        # waveforms from this distribution and computing the average
        # probability
        int_wv = (self.sample_events(nint, means=means, stds=stds) +
                  self.simulate_noise(nint, self.noise_classical))
        int_probs = self.no_photon_prob(int_wv)
        avg_probs = np.mean(int_probs, axis=0)

        # number of events seen so far
        num_events = counts.shape[0]

        # compute log likelihood for binomial distribution
        # excludes the constant term log(num_events choose counts)
        return np.sum(
            (num_events - counts) * np.log(avg_probs) +
            counts * np.log(1 - avg_probs)
        )

    def log_probability(self, theta, prior_fn, likelihood_fn, *args):
        """
        Compute the overall log-probability = log-prior + log-likelihood for the
        given parameters, likelihood function (and additional arguments).

            theta: array of parameters
            prior_fn: function for computing prior
            likelihood_fn: function for computing likelihood
            args: additional arguments for likelihood function
        """
        lp = prior_fn(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + likelihood_fn(theta, *args)

    # MCMC functions

    def _init_walkers(self, true_values, nwalkers, ndim, priors):
        """
        Initialize walkers for MCMC by scattering the true values by an amount
        proportional to the prior width, up to a maximum of WALKER_STD_MAX.
            true_values: true parameter values
            nwalkers: number of walkers
            ndim: number of parameters
            spread: amount to scatter by
        """
        # compute scatter for walkers as a fraction of the prior width
        spread = np.array([
            (p[1]-p[0])*self.WALKER_STD for p in priors
        ])

        # get indices of true parameters values that are zero
        zero_inds = (true_values == 0)

        # compute relative spreads for non-zero values
        rel_spread = spread.copy()
        rel_spread[~zero_inds] /= true_values[~zero_inds]

        # scatter walkers about true values
        pos = true_values * (
            1 + self.rng.normal(size=(nwalkers, ndim)) *
            np.minimum(np.abs(rel_spread), self.WALKER_STD_MAX)
        )

        # add small scatter to zero values
        if np.any(zero_inds):
            pos[:, zero_inds] += self.rng.normal(
                size=(nwalkers, zero_inds.sum())
            ) * self.EPS

        # force initial positions to be within the priors
        pos = pos.clip(
            np.array(priors)[:, 0], np.array(priors)[:, 1]
        )

        return pos

    def _launch_mcmc_threads(self, mcmc_fn, args, show_progress=True):
        """
        Handle logic of mapping model inputs to MCMC jobs and handle logic for
        parallelizing and showing progress.
            mcmc_fn: function to use for MCMC
            args: arguments to pass to mcmc_fn
            size: size of job list to be computed
            show_progress: whether to show progress bar
        """
        # if parallelizing with Multiprocessing
        if self.parallel:
            with Pool() as pool:
                parallel_args = (mcmc_fn, args)
                # whether to show progress with tqdm
                if show_progress:
                    samplers = list(
                        tqdm(pool.imap(*parallel_args), total=len(args))
                    )
                else:
                    samplers = pool.map(*parallel_args)
        # if not parallelizing
        else:
            samplers = map(
                mcmc_fn,
                tqdm(args) if show_progress else args
            )

        return samplers

    def _strain_mc(self, args):
        """
        Helper function for parallelizing event parameter estimation. Needs to
        be top-level to be pickle-able.

            args: tuple of arguments
        """
        # unpack arguments
        wv, p0, nwalkers, ndim, nsteps = args

        # use actual parameters as initial guesses with some scatter;
        # only restriction is that amplitude is positive
        priors = [(0, np.inf)] + [(-np.inf, np.inf)]*self.ndim_p
        pos = self._init_walkers(p0, nwalkers, ndim, priors)

        # initialize sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,
            args=(self.log_prior_event, self.log_likelihood_event, wv,)
        )

        # run MCMC
        sampler.run_mcmc(pos, nsteps)

        return sampler

    def strain_mc(self, strains, p0, nwalkers=None, nsteps=None,
                  show_progress=True):
        """
        Perform MCMC to infer the parameters of the given waveforms.
            f: array of frequencies
            strain: (no. of frequencies, no. of realizations) array of strains
            noise_spectrum: noise power spectral density
            p0: initial guesses for the walkers, use e.g. actual parameters used to
                generate the strains
            waveform_fn: function that takes parameters and returns a waveform
            nwalkers: number of walkers to use in MCMC
            nsteps: number of steps to take in MCMC
            show_progress: whether to show progress bar
        """
        # use default MCMC values if not specified
        if nwalkers is None:
            nwalkers = self.DEFAULT_WALKERS
        if nsteps is None:
            nsteps = self.DEFAULT_STEPS_EVENT

        # number of parameters to infer (one for each intrinsic parameter
        # plus one for amplitude as a proxy for distance)
        ndim = self.ndim_p + 1

        # arguments needed to be passed to helper function
        subseq_args = [
            (strains[:, wv_ind], p0[:, wv_ind], nwalkers, ndim, nsteps)
            for wv_ind in range(strains.shape[1])
        ]

        samplers = self._launch_mcmc_threads(
            self._strain_mc, subseq_args, show_progress
        )

        return samplers

    def _strain_dist_mc_seq(self, args):
        """
        Helper function for parallelizing strain MCMC by performing MCMC 
        on a subsequence of events. Needs to be top-level to be pickle-able.

            args: tuple of arguments
        """
        # use Gaussian likelihood only if hd_gaussian is True
        if self.hd_gaussian:
            likelihood_fn = self.log_likelihood_hd_gaussian

            # unpack arguments
            param_means, param_stds, pos, nwalkers, ndim, nsteps = args
            likelihood_args = (param_means, param_stds, )
        else:
            likelihood_fn = self.log_likelihood_hd

            # unpack arguments
            events_posterior, wv_ind, pos, nwalkers, ndim, nsteps = args
            likelihood_args = (events_posterior, wv_ind, )

        # initialize sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,
            args=(
                self.log_prior_dist, likelihood_fn, *likelihood_args
            )
        )

        # run MCMC
        sampler.run_mcmc(pos, nsteps)

        return sampler

    def strain_dist_mc(self, event_data,
                       nwalkers=None, nsteps=None, show_progress=True):
        """
        Compute the cumulative estimate of the astrophysical distribution
        parameters (mean and standard deviation) given the parameters inferred
        for individual events (and their errors).
            event_data: either (param_means, param_stds) tuple of 
                (no. of event, no. of params) numpy arrays if hd_gaussian
                is true, otherwise the event Posteriors object
            nwalkers: number of walkers to use in MCMC
            nsteps: number of steps to take
            show_progress: whether to show progress bar
        """
        # use default MCMC values if not specified
        if nwalkers is None:
            nwalkers = self.DEFAULT_WALKERS
        if nsteps is None:
            nsteps = self.DEFAULT_STEPS_DIST

        # number of parameters to infer: two (mean and std) for each
        # intrinsic parameter; parameters in order of [means | stds]
        ndim = 2*self.ndim_p

        # use actual astrophysical means and standard deviations as initial
        # positions for walkers
        pos = np.concatenate(
            (self.param_means[self.mcmc_mask], self.param_stds[self.mcmc_mask])
        )
        pos = self._init_walkers(pos, nwalkers, ndim, self.dist_priors)

        # arguments needed to be passed to helper function
        if self.hd_gaussian:
            # only expect and need to pass posterior means/stds if Gaussian
            param_means, param_stds = event_data
            subseq_args = [
                (param_means[:wv_ind+1], param_stds[:wv_ind+1], pos,
                 nwalkers, ndim, nsteps)
                for wv_ind in range(param_means.shape[0])
            ]
        else:
            # need to pass the full posterior samples if not Gaussian
            # (omit amplitude parameter)
            subseq_args = [
                (event_data, wv_ind,
                 pos, nwalkers, ndim, nsteps)
                for wv_ind in range(event_data.shape[0])
            ]

        samplers = self._launch_mcmc_threads(
            self._strain_dist_mc_seq, subseq_args, show_progress
        )

        return samplers

    def _count_mc_seq(self, args):
        """
        Helper function for parallelizing photon count MCMC by performing MCMC 
        on a subsequence of events. Needs to be top-level to be pickle-able.

            args: tuple of arguments
        """
        # unpack arguments
        counts, pos, nwalkers, ndim, nint, nsteps = args

        # initialize sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,
            args=(
                self.log_prior_dist, self.log_likelihood_pc,
                counts, nint,
            )
        )

        # run MCMC
        sampler.run_mcmc(pos, nsteps)

        return sampler

    def count_mc(self, counts, nwalkers=None, nsteps=None, nint=100,
                 show_progress=True):
        """
        Perform MCMC to infer the parameters of the given waveforms, after they have
        been filtered by output template filters and converted to photon counts.
            counts: (no. of realizations, no. of templates) array of photon counts
            nwalkers: number of walkers to use in MCMC
            nwalkers: number of steps to use in MCMC
            show_progress: whether to show progress bar
        """
        # use default MCMC values if not specified
        if nwalkers is None:
            nwalkers = self.DEFAULT_WALKERS
        if nsteps is None:
            nsteps = self.DEFAULT_STEPS_DIST

        # number of parameters to infer: two (mean and std) for each
        # intrinsic parameter; parameters in order of [means | stds]
        ndim = 2*self.ndim_p

        # use actual astrophysical means and standard deviations as initial
        # positions for walkers
        pos = np.concatenate(
            (self.param_means[self.mcmc_mask], self.param_stds[self.mcmc_mask])
        )
        pos = self._init_walkers(pos, nwalkers, ndim, self.dist_priors)

        # do MCMC for events up to event index wv_ind

        # arguments needed to be passed to helper function
        subseq_args = [(counts[:wv_ind+1, :], pos, nwalkers, ndim, nint, nsteps)
                       for wv_ind in range(counts.shape[0])]

        samplers = self._launch_mcmc_threads(
            self._count_mc_seq, subseq_args, show_progress
        )

        return samplers

    ##############################################################
    ############### RUN MOCK DATA PIPELINE #######################
    ##############################################################

    def run_hd(self, raw_waveforms, params):
        """
        Given the simulated events (without noise), add noise and perform
        MCMC to get the astrophysical hyper-posteriors.

            raw_waveforms: np arrays of shape
            (no. of frequencies, no. of events)
            params: np arrays of shape (no. of parameters, no. of events)
            with the true parameters used to generate the waveforms
        """
        # number of events
        n = raw_waveforms.shape[1]

        # add noise to waveforms (total noise for homodyne readout;
        # classical + quantum)
        waveforms_hd = (raw_waveforms +
                        self.simulate_noise(n, self.noise_total)
                        )

        # pick out true values of parameters to be inferred (including amplitude)
        sub_params = params[[True] + self.mcmc_mask, :]

        # do event-wise MCMC on the simulate strains
        samplers = self.strain_mc(waveforms_hd, sub_params)

        # flatten samples (retain full samples even if lean)
        event_posterior = Posterior(samplers, hyper=False, lean=False)
        event_samples = np.array(event_posterior.flat_samples)

        # now can leanify event posteriors since we've grabbed the samples
        if self.lean:
            event_posterior.leanify()

        # retrieve means and stds computed in Posterior object,
        # excluding amplitude
        means = event_posterior.means[:,1:]
        stds = event_posterior.means[:,1:]

        # do MCMC on astrophysical population
        if self.hd_gaussian:
            hd_samplers = self.strain_dist_mc((means, stds))
        else:
            hd_samplers = self.strain_dist_mc(event_samples)

        return (event_posterior, Posterior(hd_samplers, lean=self.lean))

    def run_pc(self, raw_waveforms):
        """
        Given the simulated events (without noise), add noise and perform
        MCMC to get the astrophysical hyper-posteriors.

            raw_waveforms: np arrays of shape
            (no. of frequencies, no. of events)
        """
        # number of events
        n = raw_waveforms.shape[1]

        # add noise to waveforms (only classical for photon counting)
        waveforms_pc = (raw_waveforms +
                        self.simulate_noise(n, self.noise_classical)
                        )

        # simulate photon counting given the simulated strains
        photon_probs = self.no_photon_prob(waveforms_pc)
        photon_counts = self.count_photons(photon_probs)

        # do photon counting MCMC
        pc_samplers = self.count_mc(photon_counts)

        return (Posterior(pc_samplers, lean=self.lean), )

    def run(self, n, snr_sorted=False):
        """
        Simulate events with detector noise, infer distribution parameters
        with MCMC on the strain and photon counts.

            N: number of astrophysical events to simulate
            snr_sorted: whether to sort events by SNR before doing MCMC
        """
        # draw event parameters and simulate waveforms
        params, waveforms = self.sample_events(n, True)

        # sort events by distance and perform hyper-MCMC on sorted events
        snrs = self.compute_snr(waveforms)
        if snr_sorted:
            sorted_inds = np.argsort(snrs)[::-1]

            waveforms = waveforms[:, sorted_inds]
            params = params[:, sorted_inds]

        # strain MCMC
        hd_results = self.run_hd(waveforms, params)

        # photon counting MCMC
        pc_results = self.run_pc(waveforms)

        # return computed data
        return PipelineResults(self, snrs, *hd_results, *pc_results)


class Posterior:
    """
    A class representing a posterior distribution of samples.
    """
    # default discard and thin values walker chains
    # discard should be ~ a few times the autocorrelation time
    # thin should be ~ 1/2 the autocorrelation time
    # use ~80 for autocorrelation time
    # https://emcee.readthedocs.io/en/stable/tutorials/line/
    N_DISCARD = 500
    N_THIN = 50

    def __init__(self, samples=None, hyper=True, lean=False, calc_autocorr=False):
        """
        Create a Posterior object for storing MCMC chains and chain population
        states.

            samples: MCMC chains
            hyper: whether these chains are for the astrophysical distribution
                or individual events (used for plotting routines)
            lean: whether to delete the chains and only save chain
                statistics
            calc_autocorr: whether to calculate chain autocorrelation (will
                throw error if insufficient chain lengths)
        """
        if samples is not None:
            # whether this posterior is for the astrophysical hyper-parameters
            self.hyper = hyper

            # flatten chain samples
            self.samples = list(samples)
            self.flat_samples = [s.get_chain(
                discard=self.N_DISCARD, thin=self.N_THIN, flat=True
            ) for s in self.samples if s.iteration > self.N_DISCARD]

            if len(self.flat_samples) == 0:
                raise ValueError(f"Number of steps should be larger than N_DISCARD={self.N_DISCARD}")

            # compute autocorrelation times
            if calc_autocorr:
                self.autocorr_times = [s.get_autocorr_time() for s in self.samples]

            # compute mean and standard deviation from walkers for each event
            stat_shape = (len(self.flat_samples), self.flat_samples[0].shape[1])
            self.means = np.zeros(stat_shape)
            self.stds = np.zeros(stat_shape)

            # comute means/stds for each inferred parameter for each set of chains
            for i in range(len(self.flat_samples)):
                self.means[i, :] = np.mean(self.flat_samples[i], axis=0)
                self.stds[i, :] = np.std(self.flat_samples[i], axis=0)

            # remove samples for leaner save files
            self.lean = lean
            if lean:
                self.leanify()

    def leanify(self):
        """
        Make the posterior footprint lean by removing all the samples.
        """
        self.samples = None
        self.flat_samples = None
        self.lean = True

def json_default(o):
    """
    Serializing types not natively supported by JSON.

        o: object that JSON failed to serialize
    """
    # convert numpy objects
    if isinstance(o, np.int64): return int(o)
    if isinstance(o, np.float64): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()

    # convert pipeline class instances
    if isinstance(o, (DetectionPipeline, Posterior, PipelineResults)):
        return o.__dict__

    # convert gwinc IFOs to strings
    if isinstance(o, gwinc.nb.Budget):
        return o.__class__.__name__

    if isinstance(o, types.FunctionType):
        return o.__name__

    # do not save if cannot be serialized
    return None

@dataclass
class PipelineResults:
    """
    Class for storing results from a pipeline run.
    """
    pipeline: DetectionPipeline

    # list of event SNRs
    event_snrs: np.ndarray

    # individual event inferences from homodyne detection
    event_posterior: Posterior

    # hyper-posterior from homodyne detection
    hd_posterior: Posterior

    # hyper-posterior from photon counting
    pc_posterior: Posterior

    def _save_generic(self, filename, save_fn, binary, **kwargs):
        """
        Save PipelineResults object to a file with a generic serialization
        function.
            filename: path to file
            save_fn: function to use for saving
            binary: whether to save in binary format
            kwargs: additional kwargs for save_fn
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w' + ('b' if binary else '')) as f:
            save_fn(self, f, **kwargs)

    def save(self, filename):
        """
        Save PipelineResults object to a pickle file.
            filename: path to pickle file
        """
        self._save_generic(filename, pickle.dump, True)

    def save_json(self, filename):
        """
        Save PipelineResults object to a JSON file.
            filename: path to JSON file
        """
        self._save_generic(filename, json.dump, False, default=json_default)

    @staticmethod
    def load(filename):
        """
        Load contents from a pickle file into a PipelineResults object.
            filename: path to pickle file
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_json(filename):
        """
        Load contents from a JSON file into a PipelineResults object.
            filename: path to pickle file
        """
        with open(filename, 'r') as f:
            results = json.load(f)

        # build objects from dictionaries loaded via JSON
        for m in getmembers(PipelineResults):
            if m[0] == '__dataclass_fields__':
                for attr_name, attr in m[1].items():
                    attr_class = attr.type

                    # set values in posterior objects
                    if attr_class == Posterior:
                        new_posterior = Posterior()
                        for key, value in results[attr_name].items():
                            setattr(new_posterior, key, value)
                        results[attr_name] = new_posterior

                    # set values in pipeline object
                    if attr_class == DetectionPipeline:
                        results[attr_name] = DetectionPipeline(
                            **results[attr_name]
                        )
        
        return PipelineResults(**results)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            config = yaml.safe_load(file)

        p = DetectionPipeline(**config)

        results = p.run(config['event_count'], snr_sorted=config['snr_sorted'])

        results.save_json(config['output_file'])
    else:
        print("Usage: python pipeline.py config.yaml")