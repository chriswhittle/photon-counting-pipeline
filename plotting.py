from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import corner
from pycbc.waveform import get_fd_waveform
import gwinc

import pipeline

plt.rcParams.update({
    'figure.figsize': (8, 6),
    'axes.xmargin': 0,
    'font.size': 12,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ["Computer Modern Serif"],
    'axes.grid': True,
    'legend.loc': 'best',
    'lines.linewidth': 2,
    'axes.grid.which': 'both'
})

@dataclass
class Plotter:
    results: pipeline.PipelineResults

    def sample_spectrum(self, amp_adjust=10):
        """
        Plot event and noise realization amplitude spectral density.
            amp_adjust: factor by which to scale the event amplitude
        """
        p = self.results.pipeline

        # simulate waveform and add to total and classical noise realizations
        waveform = p.sample_events(1) * amp_adjust
        wv_total = waveform + p.simulate_noise(1, p.noise_total)
        wv_classical = waveform + p.simulate_noise(1, p.noise_classical)

        plt.figure()
        plt.semilogy(p.f, np.abs(waveform), label='Astrophysical strain')
        plt.semilogy(p.f, np.abs(wv_classical), label='Photon counting observed strain')
        plt.semilogy(p.f, np.abs(wv_total), label='Homodyne observed strain')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Strain ASD [Hz$^{-1/2}$]')
        plt.legend()
        plt.show()

    def event_spectrum(self, f=np.logspace(1, 3.6, 1000), amp=7.7e-26):
        """
        Plot event post-merger signal alongside the CBC waveform and the
        interferometer noise curve.
            f: frequency array
            amp: amplitude of the post-merger signal
        """
        p = self.results.pipeline

        # compute gwinc noise curves
        trace = p.detector.run(freq=f)
        noise_total = np.sqrt(trace.psd)
        noise_classical = np.sqrt(trace.psd - trace['QuantumVacuum'].psd)

        # compute example BNS waveform; use GW170817-like parameters from
        # PRL 119, 161101 (2017)
        gw = get_fd_waveform(
            approximant='IMRPhenomD',
            mass1=1.48,
            mass2=1.265,
            distance=393,
            delta_f=1,
            f_lower=1
        )
        h = gw[0].data
        gw_f = np.array(gw[0].sample_frequencies)
        gw_noise_total2 = p.detector.run(freq=gw_f[1:]).psd
        cbc_snr = np.sqrt(np.sum(np.abs(h[1:])**2 / gw_noise_total2))

        # compute post-merger waveform
        h_post = amp*p.waveform_func(f, *p.param_means[1:])
        post_snr = np.sqrt(np.sum(np.abs(h_post)**2 / noise_total**2))

        # plot the event post-merger signal and the CBC waveform
        plt.figure()
        plt.loglog(f, noise_total, label='Total Noise', color='k')
        plt.loglog(f, noise_classical, label='Classical Noise',
                 color=trace['Coating'].style['color'])
        plt.loglog(gw_f, np.abs(h),
                   label=rf'BNS CBC Waveform, $\rho={cbc_snr:.0f}$', color='g')
        plt.loglog(f, np.abs(h_post), 
                    label=rf'Post-Merger Waveform, $\rho={post_snr:.1f}$',
                    color='purple')

        plt.xlim([f[0], f[-1]])
        plt.ylim([np.min(noise_classical)*0.9, np.max(noise_total)*0.9])
        plt.legend(ncol=2, framealpha=0.95)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Strain ASD [Hz$^{-1/2}$]')

    def photon_probabilities(self, snrs=np.logspace(-0.5, 2, 100),
                             dfs=[0, 50, 100], N=5000):
        """
        Plot the probability of detecting a photon for an event as a function
        of the waveform SNR. Also compares against different templates
        (currently only with differing peak frequencies).

            snrs: array of SNRs for which photon probabilities are computed and
                plotted
            dfs: frequency shifts of the template; each template is used to
                compute a photon probability
            N: number of noise realizations used to compute probability
                intervals
        """
        p = self.results.pipeline

        # compute waveform using the distribution mean parameters
        raw_waveform = p.waveform(p.param_means[1:])[:,np.newaxis]
        unit_snr = p.compute_snr(raw_waveform, p.noise_classical)

        # repeat waveform for adding noise later
        waveform = raw_waveform.repeat(N, axis=1)

        # likewise use the mean for the templates, but introduce some frequency
        # shifts in the peak frequency
        template_params = p.param_means[np.newaxis, 1:].repeat(len(dfs), axis=0)
        template_params[:, 0] += np.array(dfs)
        # compute template waveforms from template parameters
        templates = p.waveform(template_params.T)

        plt.figure()

        # for each SNR value, compute photon probabilities
        raw_probs = np.zeros((len(snrs), len(dfs)))
        probs = np.zeros((len(snrs), len(dfs)))
        dps = np.zeros((len(snrs), len(dfs)))
        for i, snr in enumerate(snrs):
            # compute the probability of detecting a photon for each template
            # in the absence of noise
            raw_probs[i,:] = 1 - p.no_photon_prob(
                raw_waveform * snr / unit_snr, templates
            )

            # compute the probability of detecting a photon for each template
            # over many noise realizations
            sub_probs = 1 - p.no_photon_prob(
                waveform * snr / unit_snr
                + p.simulate_noise(N, p.noise_classical), templates
            )

            # get the mean of the probabilties
            probs[i,:] = np.mean(sub_probs, axis=0)

            # get the stds of the probabilities
            dps[i,:] = np.std(sub_probs, axis=0)

        # plot computed probabilities
        for i, df in enumerate(dfs):
            p = plt.loglog(snrs, raw_probs[:,i], ls='--')
            plt.loglog(snrs, probs[:,i], c=p[0].get_color(),
                       label=rf'$\Delta f={df}$ Hz')
            # use smoothed standard deviations
            dp_smooth = np.convolve(dps[:,i], np.ones(30), 'same') / 30
            plt.fill_between(
                snrs, probs[:,i] - dp_smooth, probs[:,i] + dp_smooth,
                alpha=0.2, facecolor=p[0].get_color()
            )

        plt.xlabel(r'$\rho$')
        plt.ylabel('Probability of detecting a photon')
        plt.legend(loc='best')

    def chains(self, posterior, ind=-1):
        """
        Plot the chain of samples for the astrophysical parameters at a
        particular event number.

            posterior: posterior object for which the chains are plotted
            ind: the event number for which the chains are plotted (i.e. MCMC
                performed on events up to and including this number; or
                indicating the specific event to plot)
        """
        samples = posterior.samples[ind].get_chain()

        ndim_p = samples.shape[2]//2
        ndim = samples.shape[2]

        # fetch the true astrophysical parameters
        if posterior.dist:
            mask = self.results.pipeline.mcmc_mask
            true_params = np.concatenate((
                self.results.pipeline.param_means[mask],
                self.results.pipeline.param_stds[mask]
            ))
        else:
        # get the true event parameters (converting posterior index to
        # event index)
            true_params = self.results.event_params[
                self.results.pipeline.event_mask,
                posterior.event_numbers[ind]-1
            ]

        # iterate over each parameter and plot the chains as a function of
        # MCMC steps
        fig, axes = plt.subplots(ndim, sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.axhline(true_params[i], color="r", lw=2)
            ax.set_xlim(0, len(samples))
            ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlabel("Step Number")
    
    def corner(self, posterior, ind=-1):
        """
        Corner plot of the given posterior samples at a particular event
        number.

            posterior: posterior object to be plotted
            ind: the event number for which the chains are plotted (i.e. MCMC
                performed on events up to and including this number; or
                indicating the specific event to plot)
        """
        flat_samples = posterior.flat_samples[ind]

        # fetch the true astrophysical parameters
        if posterior.dist:
            mask = self.results.pipeline.mcmc_mask
            true_params = np.concatenate((
                self.results.pipeline.param_means[mask],
                self.results.pipeline.param_stds[mask]
            ))
        else:
        # get the true event parameters (converting posterior index to
        # event index)
            true_params = self.results.event_params[
                self.results.pipeline.event_mask, 
                posterior.event_numbers[ind]-1
            ]

        corner.corner(flat_samples, truths=true_params)

    def astro_sequence(self, pc=False, x_distance=False):
        """
        Plot the inferred (mean and uncertainty) astrophysical parameters
        (mean and standard deviation) as a function of event count.

            pc: whether to use the posteriors computed using photon counting
            x_distance: whether to plot the distance instead of the event count on the
                x-axis
        """
        if pc:
            posterior = self.results.pc_posterior
        else:
            posterior = self.results.hd_posterior
        
        # ensure numpy array
        posterior.means = np.array(posterior.means)
        posterior.stds = np.array(posterior.stds)

        # get param dimension and number of events
        ndim = posterior.means.shape[1]
        if x_distance:
            xs = self.results.event_params[0,:]
        else:
            xs = np.arange(1, posterior.means.shape[0]+1)

        # fetch the true astrophysical parameters
        mask = self.results.pipeline.mcmc_mask
        true_params = np.concatenate(
            (self.results.pipeline.param_means[mask],
             self.results.pipeline.param_stds[mask])
        )

        # plot the evolution of the inferred parameters (including means and
        # stds) as a function of events detected and used for inference
        plt.figure()
        fig, axs = plt.subplots(ndim, 1, sharex=True)
        plt.subplots_adjust(hspace=0)
        for i, ax in enumerate(axs):
            ax.plot(xs, posterior.means[:, i])
            ax.fill_between(
                xs, posterior.means[:, i] - posterior.stds[:, i],
                posterior.means[:, i] + posterior.stds[:, i], alpha=0.1
            )
            ax.axhline(true_params[i], color="r", lw=1, alpha=1, ls='--')

        if x_distance:
            ax.set_xlabel('SNR')
            ax.invert_xaxis()
        else:
            ax.set_xlabel('Event count')