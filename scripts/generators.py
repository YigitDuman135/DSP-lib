import numpy as np
import numpy.typing as npt

def dirac(N: int, dtype: npt.DTypeLike = float) -> npt.NDArray:
    """
    Unit impulse: 1 at n = 0, 0 elsewhere.
 
    Parameters
    ----------
    N : int
        Number of samples.
    dtype : numpy dtype, optional
        Output data type.
 
    Returns
    -------
    numpy.ndarray
        Impulse samples, length N.
    """
    x = np.zeros(N, dtype=dtype)
    x[0] = 1
    return x

def u(N: int, dtype: npt.DTypeLike = float) -> npt.NDArray:
    """
    Unit step: 1 for n >= 0.
 
    Parameters
    ----------
    N : int
        Number of samples.
    dtype : numpy dtype, optional
        Output data type.
 
    Returns
    -------
    numpy.ndarray
        Step samples (all ones), length N.
    """
    return np.ones(N, dtype=dtype)

def ramp(N: int, dtype: npt.DTypeLike = float) -> npt.NDArray:
    """
    Unit ramp: r[n] = n for n >= 0.
 
    Parameters
    ----------
    N : int
        Number of samples.
    dtype : numpy dtype, optional
        Output data type.
 
    Returns
    -------
    numpy.ndarray
        Ramp samples 0, 1, ..., N-1.
    """
    return np.arange(N, dtype=dtype) 

def damp_sin(N: int, f0: int, fs: int, r: npt.DTypeLike = float, phi=0.0):
    """
    Damped sinusoid: r**n * cos(2*pi*f0/fs * n + phi).
 
    Parameters
    ----------
    N : int
        Number of samples.
    f0 : float
        Frequency in Hz, in [0, fs/2].
    fs : float
        Sampling frequency in Hz.
    r : float, optional
        Decay factor per sample, r >= 0 (r < 1 decays).
    phi : float, optional
        Initial phase in radians.
 
    Returns
    -------
    numpy.ndarray of float64
        Damped sinusoid samples, length N.
    """
    if r < 0:
        raise ValueError("r must be non-negative")
    if not (0 <= f0 <= fs / 2):
        raise ValueError("f0 must lie in [0, fs/2] to avoid aliasing")
    
    n = np.arange(N)
    x = np.arangecos(2 * np.pi * f0 * n / fs + phi)
    return (r**(n)) * x

def impulse_train(N: int, m: int, dtype: npt.DTypeLike = float) -> npt.NDArray:
    """
    Impulse train: 1 at n = 0, m, 2m, ..., 0 elsewhere.
 
    Parameters
    ----------
    N : int
        Number of samples.
    m : int
        Period in samples, m >= 1.
    dtype : numpy dtype, optional
        Output data type.
 
    Returns
    -------
    numpy.ndarray
        Impulse train samples, length N.
    """
    x = np.zeros(N, dtype=dtype)
    x[::m] = 1
    return x

def wave(N: int, f0: int, fs: int, A: npt.DTypeLike = float, phi=0):
    """
    Real sinusoid: A * cos(2*pi*f0/fs * n + phi).
 
    Parameters
    ----------
    N : int
        Number of samples.
    f0 : float
        Frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    A : float, optional
        Amplitude.
    phi : float, optional
        Initial phase in radians.
 
    Returns
    -------
    numpy.ndarray of float64
        Sinusoid samples, length N.
    """

    n = np.arange(N)
    w0 = 2 * np.pi * (f0 / fs)
    return A * np.cos(w0 * n + phi)

def rexp(N: int, A: npt.DTypeLike = float):
    """
    Real exponential: A**n for n >= 0.
 
    Parameters
    ----------
    N : int
        Number of samples.
    A : float
        Base (|A| < 1 decays, |A| > 1 grows).
 
    Returns
    -------
    numpy.ndarray of float64
        Exponential samples, length N.
    """
    n = np.arange(N)
    return np.float64(A) ** n

def cexp(N: int, f0: int, fs: int, A: npt.DTypeLike = float, phi=0):
    """
    Complex exponential: A * exp(j*(2*pi*f0/fs * n + phi)).
 
    Parameters
    ----------
    N : int
        Number of samples.
    f0 : float
        Frequency in Hz (negative rotates clockwise).
    fs : float
        Sampling frequency in Hz.
    A : float, optional
        Amplitude.
    phi : float, optional
        Initial phase in radians.
 
    Returns
    -------
    numpy.ndarray of complex128
        Complex exponential samples, length N.
    """

    n = np.arange(N)
    w0 = 2 * np.pi * (f0 / fs)
    return A * np.cos(w0 * n + phi) + 1j * A * np.sin(w0 * n + phi)

def lin_chirp(f0: int, f1: int, fs: int, T: int, phi=0):
    """
    Real linear chirp sweeping from f0 to f1.
 
    Parameters
    ----------
    f0 : float
        Start frequency in Hz, in [0, fs/2).
    f1 : float
        End frequency in Hz, in [0, fs/2).
    fs : float
        Sampling frequency in Hz.
    T : float
        Duration in seconds; N = round(T * fs).
    phi : float, optional
        Initial phase in radians.
 
    Returns
    -------
    numpy.ndarray of float64
        Chirp samples, length N.
    """

    N = round(T * fs)
    if N < 2:
        raise ValueError("chirp needs at least 2 samples")
    if not (0 <= f0 < fs / 2 and 0 <= f1 < fs / 2):
        raise ValueError("f0 and f1 must lie in [0, fs/2) to avoid aliasing")
    
    k = (f1 - f0) * fs / (N - 1)

    theta = phi
    x = np.zeros(N)

    for n in range(N):
        x[n] = np.cos(theta)
        theta += (2 * np.pi / fs) * (f0 + (k / fs) * (n + 0.5) )
    
    return x

def square(N: int, f0: int, fs: int, phi=0):
    """
    Band-limited square wave: odd harmonics below fs/2 only.
 
    Parameters
    ----------
    N : int
        Number of samples.
    f0 : float
        Fundamental frequency in Hz, in (0, fs/2).
    fs : float
        Sampling frequency in Hz.
    phi : float, optional
        Initial phase in radians.
 
    Returns
    -------
    numpy.ndarray of float64
        Square wave samples, length N.
    """

    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must lie in (0, fs/2)")

    n = np.arange(N)
    x = np.zeros(N)
    k = 1
    
    while k * 2 * f0 < fs:
        x += np.sin(2 * np.pi * k * f0 / fs * n + k * phi) / k
        k += 2
    
    return (4 / np.pi) * x

def sawtooth(N: int, f0: int, fs: int, phi=0.0):
    """
    Band-limited rising sawtooth: all harmonics below fs/2.
 
    Crosses zero at n = 0; phi = pi matches scipy.signal.sawtooth.
 
    Parameters
    ----------
    N : int
        Number of samples.
    f0 : float
        Fundamental frequency in Hz, in (0, fs/2).
    fs : float
        Sampling frequency in Hz.
    phi : float, optional
        Initial phase in radians.
 
    Returns
    -------
    numpy.ndarray of float64
        Sawtooth samples, length N.
    """
    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must lie in (0, fs/2)")

    n = np.arange(N)
    x = np.zeros(N)
    k = 1
    while 2 * k * f0 < fs:
        x += (-1) ** (k + 1) * np.sin(2 * np.pi * k * f0 / fs * n + k * phi) / k
        k += 1
    return (2 / np.pi) * x