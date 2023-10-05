from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

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
    'lines.linewidth': 2
})

@dataclass
class Plotter:
    results: pipeline.PipelineResults

    def plot_astro_chains(self, ind=-1):
        """
        Plot the chain of samples for the astrophysical parameters at a
        particular event number.

            ind: the event number for which the chains are plotted (i.e. MCMC
                performed on events up to and including this number)
        """
        samples = self.results.astro_samples[ind].get_chain()

        ndim_p = samples.shape[2]//2
        ndim = samples.shape[2]

        # fetch the true astrophysical parameters
        true_params = np.concatenate(
            (self.results.config.param_means, self.results.config.param_stds)
        )

        # iterate over each parameter and plot the chains as a function of
        # MCMC steps
        fig, axes = plt.subplots(ndim, sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.axhline(true_params[i], color="r", lw=2)
            ax.set_xlim(0, len(samples))
            ax.yaxis.set_label_coords(-0.1, 0.5)

    def plot_astro_sequence(self, pc=False):
        """
        Plot the inferred (mean and uncertainty) astrophysical parameters
        (mean and standard deviation) as a function of event count.

            pc: whether to use the posteriors computed using photon counting
        """
        if pc:
            posterior = self.results.pc_astro_flat_samples
        else:
            posterior = self.results.astro_flat_samples

        # evaluate the mean and stds for the posteriors of the inferred
        # astrophysical parameters
        posterior_means = np.array([p.mean(axis=0) for p in posterior])
        posterior_stds = np.array([p.std(axis=0) for p in posterior])

        ndim = posterior_means.shape[1]
        event_counts = np.arange(1, posterior_means.shape[0]+1)

        # fetch the true astrophysical parameters
        true_params = np.concatenate(
            (self.results.config.param_means, self.results.config.param_stds)
        )

        # plot the evolution of the inferred parameters (including means and
        # stds) as a function of events detected and used for inference
        plt.figure()
        fig, axs = plt.subplots(ndim, 1, sharex=True)
        plt.subplots_adjust(hspace=0)
        for i, ax in enumerate(axs):
            ax.plot(event_counts, posterior_means[:, i])
            ax.fill_between(
                event_counts, posterior_means[:, i] - posterior_stds[:, i],
                posterior_means[:, i] + posterior_stds[:, i], alpha=0.1
            )
            ax.axhline(true_params[i], color="r", lw=1, alpha=1, ls='--')
        
        ax.set_xlabel('Event count')