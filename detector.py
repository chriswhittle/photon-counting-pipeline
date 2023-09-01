import numpy as np
import gwinc
import waveform

SPEED_OF_LIGHT = 299792458 # m/s
REDUCED_PLANCK = 1.0545718e-34 # m^2 kg / s

def fetch_ifo_psd(f, budget):
    """
    Use gwinc to compute classical and quantum noises for a given
    interferometer:
        f: array of frequencies
        budget: gwinc budget object, e.g. gwinc.load_budget(ifo) where ifo is
            one of 'aLIGO', 'CE1' etc.
    Returns PSDs for total and classical noises.
    """
    trace = budget.run(freq=f)

    quantum = trace['QuantumVacuum'].psd
    classical = trace.psd - quantum

    return trace.psd, classical

def simulate_noise(f, N, psd):
    """
    Generate N realizations of noise at the specified frequencies shaped by
    the given power spectral density.
        f: array of frequencies
        N: number of realizations to generate
        psd: noise power spectral density
    """
    # simulate white noise with random phase (normalized to have unit standard
    # deviation in magnitude)
    white_noise = (np.random.normal(size=(len(f), N))
                   + 1j*np.random.normal(size=(len(f), N))) / np.sqrt(2)
    
    # scale noise by PSD
    white_noise *= np.sqrt(psd[:, np.newaxis])

    return white_noise

def no_photon_prob(f, strain, budget, templates):
    """
    Given a series of strain spectra, convert to output optical power and then
    subsequently to "probability of observing zero photos" in each optical mode
    at the output.
        f: array of frequencies
        strain: (no. of frequencies) x (no. of realizations) array of strain
        budget: gwinc budget object, e.g. gwinc.load_budget(ifo) where ifo is
            one of 'aLIGO', 'CE1' etc.
        templates: (no. of frequencies) x (no. of templates) array of templates
    Returns matrix (no. of realizations) x (no. of templates) of probabilities
    """
    # TODO: this assumes transfer function is flat, i.e. (32)
    # neglects any recycling in IFO.
    Parm = gwinc.ifo.noises.ifo_power(budget.ifo).parm
    Larm = budget.ifo.Infrastructure.Length
    wavelength = budget.ifo.Laser.Wavelength
    wavenumber = 2 * np.pi / wavelength
    omega = SPEED_OF_LIGHT * wavenumber

    # change in length -> quanta at output; eq. (32) in McCuller22
    # TODO: make frequency-dependent array, np.repeat
    g = 2 * wavenumber * np.sqrt(Parm / (2*REDUCED_PLANCK*omega))

    # output fields
    output_fields = g * (strain * Larm)

    # mean fields in output templates
    # units of sqrt(quanta) TODO: check this
    mean_counts = np.zeros(
        (strain.shape[1], templates.shape[1]), dtype=np.complex
    )
    for i, template in enumerate(templates.T):
        mean_counts[:, i] = waveform.inner_product(
            f,
            template[:,np.newaxis].repeat(strain.shape[1], axis=1),
            output_fields
        )
    
    # convert mean counts to probability of zero photons
    # Pr(n=0) = |<0|alpha>|^2 = exp(-|alpha|^2)
    return np.exp(-np.abs(mean_counts)**2)

def count_photons(prob):
    """
    Given a matrix of probabilities of observing zero photons in each 
    realization for each template mode, randomly populate each with 0 or 1
    photons.
        prob: (no. of realizations) x (no. of templates) matrix of probabilities
    Returns a matrix of the same shape with 0s and 1s.
    """
    return np.array(np.random.random(prob.shape) > prob, dtype=np.int)