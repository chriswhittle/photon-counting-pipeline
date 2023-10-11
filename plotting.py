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
        h_post = amp*p.waveform_func(f, *p.param_means)
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
        if posterior.hyper:
            true_params = np.concatenate((
                self.results.pipeline.param_means,
                self.results.pipeline.param_stds
            ))

        # iterate over each parameter and plot the chains as a function of
        # MCMC steps
        fig, axes = plt.subplots(ndim, sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            if posterior.hyper:
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
        if posterior.hyper:
            true_params = np.concatenate((
                self.results.pipeline.param_means,
                self.results.pipeline.param_stds
            ))

        corner.corner(
            flat_samples, truths=true_params if posterior.hyper else None
        )

    def astro_sequence(self, pc=False):
        """
        Plot the inferred (mean and uncertainty) astrophysical parameters
        (mean and standard deviation) as a function of event count.

            pc: whether to use the posteriors computed using photon counting
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
        event_counts = np.arange(1, posterior.means.shape[0]+1)

        # fetch the true astrophysical parameters
        true_params = np.concatenate(
            (self.results.pipeline.param_means, self.results.pipeline.param_stds)
        )

        # plot the evolution of the inferred parameters (including means and
        # stds) as a function of events detected and used for inference
        plt.figure()
        fig, axs = plt.subplots(ndim, 1, sharex=True)
        plt.subplots_adjust(hspace=0)
        for i, ax in enumerate(axs):
            ax.plot(event_counts, posterior.means[:, i])
            ax.fill_between(
                event_counts, posterior.means[:, i] - posterior.stds[:, i],
                posterior.means[:, i] + posterior.stds[:, i], alpha=0.1
            )
            ax.axhline(true_params[i], color="r", lw=1, alpha=1, ls='--')
        
        ax.set_xlabel('Event count')