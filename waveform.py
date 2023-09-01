import numpy as np

def frequency_bins(sampling_frequency, bin_width):
    """
    Generate one-sided list of frequencies with given:
        sampling_frequency
        bin_width
    """
    return np.arange(bin_width, sampling_frequency/2, bin_width)

def draw_params(N, *args):
    """
    Draw parameter values from Gaussian distributions with given means and
    standard deviations.

        N: number of samples to draw
        args: tuples of the form (mean, std) for each parameter
    
    Returns a numpy array of shape (len(args), N) containing the drawn values.
    """
    samples = [np.random.normal(mean, std, size=N) for mean, std in args]
    return np.array(samples)

def inner_product(f, a, b):
    """
    Compute the inner product of two signals a and b, given the list of
    frequencies. Equivalent to a time domain inner product by Parseval's
    Theorem.
    """
    bw = f[1] - f[0]
    return bw * np.sum(a * np.conj(b), axis=0) / (2*np.pi)

def waveform(f, A, f0, bw, phi, lim_factor=30):
    """
    Compute and return a Lorentzian wavelet with the following parameters,
    each stored in numpy arrays of equal length (aside from f):
        f: numpy array of equally-spaced frequency bins
        A: amplitude
        f0: peak frequency
        bw: bandwidth (FWHM)
        phi: phase of wavelet at peak frequency
    """
    # duplicate f across a new axis for broadcasting
    # if A is a numpy array
    if isinstance(A, np.ndarray):
        f = f[:, np.newaxis].repeat(A.shape[0], axis=1)

    # Lorentzian wavelet # TODO: update waveform
    wavelet = np.exp(1j * phi) / (1 + ((f - f0) / (bw/2))**2)

    #TODO: add time delay parameter; ~ exp(-2j*pi*f*t0)

    # hard cutoff outside several bandwidths
    wavelet *= (np.abs(f - f0) < lim_factor * bw)

    # time delay to center wavelet at t=0
    bw = f[1] - f[0]
    wavelet *= np.exp(-2j * np.pi * f  / (2*bw))

    # normalize wavelet
    wavelet /= np.sqrt(inner_product(f, wavelet, wavelet))

    # scale wavelet by amplitude
    wavelet *= A

    return wavelet