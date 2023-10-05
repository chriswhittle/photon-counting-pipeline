import numpy as np
import numexpr as ne

def exp(x):
    """
    Compute exp(x) using numexpr for speed.
    """
    return ne.evaluate('exp(x)')

def lorentzian(f, f0, bw, phi, t0):
    """
    Compute and return a unit-amplitude Lorentzian wavelet with the
    following parameters, each stored in numpy arrays of equal length (aside 
    from f):
        f: numpy array of equally-spaced frequency bins
        f0: peak frequency
        bw: bandwidth (FWHM)
        phi: phase of wavelet at peak frequency
        t0: time delay
    """
    bin_width = f[1] - f[0]

    # Lorentzian wavelet
    wavelet = exp(1j * phi) / (1 + ((f - f0) / (bw/2))**2)

    # wavelet time delay (such that wavelet is centered at t=0 for t0=0)
    wavelet *= exp(-2j * np.pi * f  * (t0 + 1/(2*bin_width)))

    return wavelet