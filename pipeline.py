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
                 snr_cutoff=1, distance_uncertainty=0.03, distance_prior=True,
                 param_means=None, param_stds=None, mcmc_mask=None,
                 dist_priors=None, dist_gaussian=False, template_params=None,
                 parallel=True, checkpoints=4, **kwargs):
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
            snr_cutoff: SNR threshold for detection of an event (relative to
                classical noise); post-merger waveforms will be drawn from a
                distribution that uses this as a minimum (default is 1)
            distance_uncertainty: relative uncertainty in the distance
                inference from the coalescence signal, used as the prior
                for the post-merger signal inference (requires distance_prior
                to be True)
            distance_prior: whether to use the true distance as a prior
                for inference on a post-merger event (e.g. having inferred
                it from the coalescence)
            param_means: list of means for waveform parameters; length should be
                equal to 1 + number of arguments of waveform_func (first is
                reserved for an amplitude parameter)
            param_stds: list of standard deviations for waveform parameters;
                length same as param_means
            mcmc_mask: list of booleans indicating which parameters to infer;
                length same as param_means
            dist_priors: list of tuples of (lower, upper) bounds for each
                inferred parameter for the astrophysical distribution
                in the format [mean_lims | std_lims] =
                [(mean_lower, mean_upper), ..., (std_lower, std_upper), ...]
            dist_gaussian: whether to assume Gaussian posteriors for the
                event parameters when computing the astrophysical posterior
                (only appropriate for events above threshold that do not have
                uniform posteriors)
            template_params: list of parameters for each template waveform for
                photon counting detection; length should equal signature of
                waveform_func (i.e. no amplitude parameter)
            parallel: whether to use Multiprocessing parallelization
            checkpoints: how often to save posteriors (1 = save all posteriors)
        """
        # seeded random number generator to use throughout
        self.rng = np.random.default_rng(0)

        # generate frequency bins
        self.bw, self.f_low, self.f_high, self.f_sample = (
            bin_width, f_low, f_high, f_sample
        )
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

        # whether to use the true distance for the prior in event inference
        self.distance_prior = distance_prior
        # relative distance uncertainty; used for the distance prior
        # on event parameter inference
        self.distance_uncertainty = distance_uncertainty

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
        self.dist_gaussian = dist_gaussian
        if self.dist_gaussian and self.snr_cutoff < 1:
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

        # how frequently to save posteriors
        self.checkpoints = checkpoints

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

    def compute_snr(self, wv, noise_psd=None):
        """
        Compute the SNR of the given waveform:
        SNR = sqrt(2 * integral(|waveform|^2 / noise))
            wv: waveform (frequency, event)
            noise_psd: noise PSD to use for SNR computation, if None defaults
                to IFO noise_total
        """
        if noise_psd is None:
            noise_psd = self.noise_total

        # duplicate noise across a new axis for broadcasting
        # if params is 2-dimensional
        wv = np.array(wv)
        if len(wv.shape) == 2:
            noise = noise_psd[:, np.newaxis].repeat(wv.shape[1], axis=1)
        else:
            noise = noise_psd
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
        # (otherwise use the provided means and stds)
        if means is None:
            means = self.param_means
        if stds is None:
            stds = self.param_stds

        # SNR ~ amplitude ~ 1/distance
        # distance CDF = (distance/distance_max)^3
        # where distance_max corresponds to the provided SNR cutoff;
        # SNR = SNR_cutoff * (distance_max/distance)
        # use dimensionless distance i.e. set distance_max = 1
        distance_samples = (self.rng.random(N))**(1/3)
        snr_samples = self.snr_cutoff / distance_samples

        # draw parameters from Gaussian distributions (including amplitude
        # as the first parameter)
        param_samples = self.draw_params(N, means, stds)
        # compute waveforms using the sampled parameter values; exclude
        # amplitude parameter
        waveforms = self.waveform(param_samples[1:,:])

        # compute SNR of waveforms to renormalize (since waveforms of unit
        # normalization will have varying SNRs)
        unit_snrs = self.compute_snr(waveforms, self.noise_classical)
        amplitudes = snr_samples / unit_snrs
        # modify waveforms to take into account extrinsic ampplitude modifier
        # (i.e. distance from source)
        waveforms *= amplitudes

        # finally, multiply again by the intrinsic amplitude modifier
        waveforms *= param_samples[0,:]

        # return waveforms and parameters if requested
        if return_params:
            combined_params = np.vstack((distance_samples, param_samples))
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
            prob: (no. of templates, no. of realizations) matrix of probabilities
        Returns a matrix of the same shape with 0s and 1s.
        """
        return np.array(self.rng.random(prob.shape) > prob, dtype=int).T

    ##############################################################
    ###################### MCMC METHODS ##########################
    ##############################################################

    # default parameters for MCMC
    DEFAULT_WALKERS = 50
    DEFAULT_STEPS_EVENT = 1500
    DEFAULT_STEPS_DIST = 2000

    # number of noise realizations to use when integrating probability
    # of detecting a photon given waveform parameters
    DEFAULT_PHOTON_INT = 10000

    # constants for setting up walker initial positions
    WALKER_STD = 3e-1
    WALKER_STD_MAX = 1e-1
    EPS = 1e-3

    # prior, likelihood, probability functions
    # these need to be top-level to be pickle-able by Multiprocessing

    def log_prior_event(self, theta, distance):
        """
        Compute the log-prior for the given parameters of a single event, the
        result of integrating a Gaussian over uniform priors on the mean
        and standard deviation.
            theta: array of parameters
            distance: true distance of event
        """
        # distance should always be greater than 0
        # (need not be less than 1 if true distance is given)
        if theta[0] < 0:
            return -np.inf

        # if we have knowledge of the event distance from the coalescence,
        # use Gaussian uncertainty (math.log for better scalar performance)
        if self.distance_prior:
            distance_std = distance * self.distance_uncertainty
            distance_prior = -math.log(distance_std)
            distance_prior -= (theta[0] - distance)/2/distance_std**2
        # if we have no knowledge of the distance,
        # f(d) = 3 * (distance / max distance)^2
        else:
            # should enforce maximum distance if we have no other distance info
            if theta[0] > 1:
                return -np.inf
            distance_prior = 2 * math.log(theta[0])

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

        # if prior very small
        if prior <= 0:
            return -np.inf
        # use math.log for scalar performance
        return math.log(prior) + distance_prior

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

    def _compute_walker_waveform(self, theta):
        """
        Compute the waveform given parameters from MCMC walkers.

            theta: array of parameters
        """
        # use distribution mean values for parameters that are not being
        # inferred
        params = self.param_means.copy()
        params[self.mcmc_mask] = theta[1:]

        # compute waveform using all parameters except the amplitude
        model = self.waveform(params[1:])
        # rescale the waveform by the distance to the source
        # (extrinsic amplitude)
        unit_snr = self.compute_snr(model, self.noise_classical)
        model *= self.snr_cutoff / theta[0] / unit_snr
        # rescale by the intrinsic amplitude
        model *= theta[1]

        return model

    def log_likelihood_event_hd(self, theta, strain):
        """
        Log-likelihood function using Gaussian noise, e.g. (5) from
        PASA vol. 36 e10.
            theta: array of parameters
            strain: strain waveform
        """
        model = self._compute_walker_waveform(theta)

        # compute log likelihood for Gaussian noise
        return -0.5 * np.sum(
            np.abs(strain - model) ** 2 / self.noise_total
            + self.log_noise_total
        )

    def log_likelihood_event_pc(self, theta, counts):
        """
        Log-likelihood function for photon counting for a single event.
            theta: array of event parameters
            counts: array of photon counts (0 or 1 for each template)
        """
        model = self._compute_walker_waveform(theta)

        # for these waveform parameters, compute the probability
        # of observing zero photons in each template mode by sampling
        # noise realizations and computing the average probability
        int_wv = (
            model[:,np.newaxis].repeat(self.DEFAULT_PHOTON_INT, axis=1) +
            self.simulate_noise(
                self.DEFAULT_PHOTON_INT, self.noise_classical
        ))
        int_probs = self.no_photon_prob(int_wv)
        avg_probs = np.mean(int_probs, axis=0)

        # if we saw counts where this waveform should not produce any,
        # it is impossible (and vice versa)
        if any(counts & (avg_probs == 0)) or any((~counts) & (avg_probs == 1)):
            return -np.inf
        # compute log likelihood: 1-(no photon prob) where we saw counts
        # and (no photon prob) where we did not
        all_probs = np.vstack((avg_probs, 1-avg_probs))
        return np.sum(
            np.log(all_probs[counts, np.arange(all_probs.shape[1])])
        )

    def log_likelihood_dist(self, theta, event_post, wv_ind, nint=10000):
        """
        Log-likelihood function for hyper-parameter MCMC,
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

    def log_likelihood_dist_gaussian(self, theta, param_means, param_stds):
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

    def log_probability(self, theta, prior_fn, likelihood_fn,
                        prior_args=[], likelihood_args=[]):
        """
        Compute the overall log-probability = log-prior + log-likelihood for the
        given parameters, likelihood function (and additional arguments).

            theta: array of parameters
            prior_fn: function for computing prior
            likelihood_fn: function for computing likelihood
            prior_args: additional arguments for prior function
            likelihood_args: additional arguments for likelihood
        """
        lp = prior_fn(theta, *prior_args)
        if not np.isfinite(lp):
            return -np.inf
        return lp + likelihood_fn(theta, *likelihood_args)

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

    def _event_mc(self, args):
        """
        Helper function for parallelizing event parameter estimation. Needs to
        be top-level to be pickle-able.

            args: tuple of arguments
        """
        # unpack arguments
        likelihood_fn, wv, p0, nwalkers, ndim, nsteps = args

        # set up initial walkers:
        # for distance, use points near true distance if given distance info
        if self.distance_prior:
            priors = [
                (p0[0]*(1 - self.distance_uncertainty), 
                 p0[0]*(1 + self.distance_uncertainty))
            ]
        # otherwise pick any distance up to maximum (1 in these units)
        else:
            priors = [(0, 1)]
        # for remaining parameters, use true values as initial guesses
        # with some scatter
        priors = priors + [(-np.inf, np.inf)]*self.ndim_p
        pos = self._init_walkers(p0, nwalkers, ndim, priors)

        # initialize sampler using arguments: event-specific prior function,
        # event-specific likelihood function, true distances and strain
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,
            args=(
                self.log_prior_event, likelihood_fn, [p0[0]], [wv]
            )
        )

        # run MCMC
        sampler.run_mcmc(pos, nsteps)

        return sampler

    def event_mc(self, likelihood_fn, data, p0, nwalkers=None, nsteps=None,
                  show_progress=True):
        """
        Perform MCMC to infer the parameters of the given data.
            likelihood_fn: function used to evaluate the likelihood of detecting
                the given data
            data: data on which we perform MCMC, for homodyne detection;
                - strain (no. of frequencies, no. of realizations)
                - photon counting (no. of templates, no. of realizations)
            p0: initial guesses for the walkers, use e.g. actual parameters used
                to generate the strains
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
        # plus one for distance)
        ndim = self.ndim_p + 1

        # arguments needed to be passed to helper function
        subseq_args = [
            (
                likelihood_fn, data[:, wv_ind], p0[:, wv_ind],
                nwalkers, ndim, nsteps
            )
            for wv_ind in range(data.shape[1])
        ]

        samplers = self._launch_mcmc_threads(
            self._event_mc, subseq_args, show_progress
        )

        return samplers

    def _dist_mc_seq(self, args):
        """
        Helper function for parallelizing strain MCMC by performing MCMC 
        on a subsequence of events. Needs to be top-level to be pickle-able.

            args: tuple of arguments
        """
        # use Gaussian likelihood only if dist_gaussian is True
        if self.dist_gaussian:
            likelihood_fn = self.log_likelihood_dist_gaussian

            # unpack arguments
            param_means, param_stds, pos, nwalkers, ndim, nsteps = args
            likelihood_args = (param_means, param_stds, )
        else:
            likelihood_fn = self.log_likelihood_dist

            # unpack arguments
            events_posterior, wv_ind, pos, nwalkers, ndim, nsteps = args
            likelihood_args = (events_posterior, wv_ind, )

        # initialize sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,
            args=(
                self.log_prior_dist, likelihood_fn, [], likelihood_args
            )
        )

        # run MCMC
        sampler.run_mcmc(pos, nsteps)

        return sampler

    def dist_mc(self, event_data,
                       nwalkers=None, nsteps=None, show_progress=True):
        """
        Compute the cumulative estimate of the astrophysical distribution
        parameters (mean and standard deviation) given the parameters inferred
        for individual events (and their errors).
            event_data: either (param_means, param_stds) tuple of 
                (no. of event, no. of params) numpy arrays if dist_gaussian
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
        if self.dist_gaussian:
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
            self._dist_mc_seq, subseq_args, show_progress
        )

        return samplers

    ##############################################################
    ############### RUN MOCK DATA PIPELINE #######################
    ##############################################################

    def run_pipeline(self, mcmc_fn, raw_waveforms, noise, params):
        """
        Given the simulated events (without noise), add noise, process the
        waveform with detector physics then do MCMC to get the event-wise and
        then astrophysical hyper-posteriors.

            mcmc_fn: function that handles the (noisy) waveforms and produces
                event-wise posteriors
            raw_waveforms: np arrays of shape
                (no. of frequencies, no. of events)
            noise: noisd PSD to use to simulate noise and add to waveform
                before processing by mcmc_fn
            params: np arrays of shape (no. of parameters, no. of events)
            with the true parameters used to generate the waveforms
        """
        # number of events
        n = raw_waveforms.shape[1]

        # add noise to waveforms
        waveforms = (raw_waveforms + self.simulate_noise(n, noise))

        # pick out true values of parameters to be inferred
        # (including amplitude)
        sub_params = params[[True] + self.mcmc_mask, :]

        # simulate detector physics and do MCMC to get
        # samplers
        samplers = mcmc_fn(waveforms, sub_params)

        # flatten samples (retain full samples)
        event_posterior = Posterior(samplers, hyper=False)
        event_samples = np.array(event_posterior.flat_samples)

        # now can leanify event posteriors since we've grabbed the samples
        event_posterior.leanify(self.checkpoints)

        # retrieve means and stds computed in Posterior object,
        # excluding amplitude
        means = event_posterior.means[:,1:]
        stds = event_posterior.means[:,1:]

        # do MCMC on astrophysical population
        if self.dist_gaussian:
            dist_samplers = self.dist_mc((means, stds))
        else:
            dist_samplers = self.dist_mc(event_samples)

        return (
            event_posterior,
            Posterior(dist_samplers, checkpoints=self.checkpoints)
        )

    def run_hd(self, raw_waveforms, params):
        """
        Run standard homodyne detection pipeline with MCMC on strain (quantum
        and classical noise).

            raw_waveforms: np arrays of shape
            (no. of frequencies, no. of events)
            params: np arrays of shape (no. of parameters, no. of events)
            with the true parameters used to generate the waveforms
        """
        def hd_mc(waveforms, sub_params):
            # do event-wise MCMC on the strain directly
            samplers = self.event_mc(
                self.log_likelihood_event_hd, waveforms, sub_params
            )

            return samplers
        return self.run_pipeline(
            hd_mc, raw_waveforms, self.noise_total, params
        )

    def run_pc(self, raw_waveforms, params):
        """
        Run photon counting detection pipeline with MCMC on photon counts (just
        classical noise).

            raw_waveforms: np arrays of shape
            (no. of frequencies, no. of events)
            params: np arrays of shape (no. of parameters, no. of events)
            with the true parameters used to generate the waveforms
        """
        def pc_mc(waveforms, sub_params):
            # simulate photon counting given the simulated strains
            photon_probs = self.no_photon_prob(waveforms)
            photon_counts = self.count_photons(photon_probs)

            # do event-wise MCMC on the detected photon counts given
            # distance information
            samplers = self.event_mc(
                self.log_likelihood_event_pc, photon_counts, sub_params
            )

            return samplers

        return self.run_pipeline(
            pc_mc, raw_waveforms, self.noise_classical, params
        )

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
        distances = params[0,:]
        if snr_sorted:
            sorted_inds = np.argsort(distances)

            distances = distances[sorted_inds]
            waveforms = waveforms[:, sorted_inds]
            params = params[:, sorted_inds]

        # strain MCMC
        hd_results = self.run_hd(waveforms, params)

        # photon counting MCMC
        pc_results = self.run_pc(waveforms, params)

        # return computed data
        return PipelineResults(self, distances, *hd_results, *pc_results)


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

    def __init__(self, samples=None, hyper=True, checkpoints=1, calc_autocorr=False):
        """
        Create a Posterior object for storing MCMC chains and chain population
        states.

            samples: MCMC chains
            hyper: whether these chains are for the astrophysical distribution
                or individual events (used for plotting routines)
            checkpoints: how frequently to keep posteriors (=1 means keep
                all posteriors)
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
            self.checkpoints = checkpoints
            self.leanify(checkpoints)

    def leanify(self, checkpoints=0):
        """
        Make the posterior footprint lean by removing all samples except
        multiples of the checkpoint frequency.

            checkpoints: how regularly to save the samples
        """
        self.checkpoints = checkpoints

        # do nothing if we need to keep all samples
        if checkpoints == 1:
            return

        # delete all sam
        if checkpoints == 0:
            self.samples = []
            self.flat_samples = []

        self.samples = self.samples[checkpoints::checkpoints]
        self.flat_samples = self.samples[checkpoints::checkpoints]

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

    # list of event distances
    event_distances: np.ndarray

    # individual event inferences from homodyne detection
    hd_event_posterior: Posterior

    # individual event inferences from homodyne detection
    pc_event_posterior: Posterior

    # distribution posterior from homodyne detection
    hd_posterior: Posterior

    # distribution posterior from photon counting
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
        # load config file given as first command-line argument
        with open(sys.argv[1], 'r') as file:
            config = yaml.safe_load(file)

        # initialize pipeline object
        p = DetectionPipeline(**config)

        # run full pipeline and save output to JSON file
        results = p.run(config['event_count'], snr_sorted=config['snr_sorted'])
        results.save_json(config['output_file'])
    else:
        print("Usage: python pipeline.py config.yaml")