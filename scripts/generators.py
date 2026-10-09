import numpy as np

def dirac(N, n0=0, dtype=float):
    """
    Generate the unit impulse signal with shift n0: δ[n - n0].
    
    Parameters
    ----------
    N : int
        Number of samples in the unit impulse function
    n0 : int, optional
        offset used to determine the position of sample with amplitude 1

    When n0 >= 0: starting index that n0 offsets is 0
         n0  < 0: starting index that n0 offsets is N - 1

    dtype : numpy data type, optional
        Data type of the numpy array

    Returns
    -------
    numpy array with data type dtype
        unit impulse as numpy dtype array named x
    """
    x = np.zeros(N, dtype)
    
    if n0 >= 0:
        x[n0] = 1
    else:
        x[N + n0] = 1

    return x

def u(N, n0=0, dtype=float):
    """
    Generate the unit step signal with shift n0: u[n - n0].
    
    Parameters
    ----------
    N : int
        Number of samples in the unit step function
    n0 : int, optional
        offset used to determine the position of intial sample with amplitude 1

    When n0 >= 0: starting index that n0 offsets is 0
         n0  < 0: starting index that n0 offsets is N - 1

    dtype : numpy data type, optional
        Data type of the numpy array

    Returns
    -------
    numpy array with data type dtype
        unit step as numpy dtype array named x
    """
    x = np.zeros(N, dtype)
    
    if n0 >= 0:
        x[n0::] = 1
    else:
        x[:(N + n0)] = 1

    return x

def wave(N, A, f0, fs, phi=0):
    """
    Generate a real cosine wave: A * cos(2*pi*f0/fs * n + phi).

    Parameters
    ----------
    N : int
        Length of the generated wave
    A : int
        Amplitude of wave
    f0 : int
        Frequency of the wave
    fs : int
        Frequency at which the wave was sampled
    phi : float, optional
        Phase of the wave in radians

    Returns
    -------
    numpy.ndarray of float64
        The wave samples, length N.
    """

    n = np.arange(N)

    w0 = 2 * np.pi * (f0 / fs)

    x = A * np.cos(w0 * n + phi)

    return x

def cexp(N, A, f0, fs, phi=0):
    """
    Generate a complex exponential: A * exp(j * (2*pi*f0/fs * n + phi)).
    
    Parameters
    ----------
    N : int
        Length of the generated complex exponential
    A : int
        Amplitude of complex exponential
    f0 : int
        Frequency of the complex exponential
    fs : int
        Frequency at which the complex exponential was sampled
    phi : float, optional
        Phase of the complex exponential in radians

    Returns
    -------
    numpy.ndarray of complex128
        The complex exponential samples, length N.
    """

    n = np.arange(N)

    w0 = 2 * np.pi * (f0 / fs)

    x = A * np.cos(w0 * n + phi) + 1j * A * np.sin(w0 * n + phi)

    return x

def lin_chirp(f0, f1, fs, T, phi=0):
    """
    Generate a real linear chirp sweeping from f0 to f1.

    Parameters
    ----------
    f0 : float
        Start frequency in Hz (at sample 0).
    f1 : float
        End frequency in Hz (at the last sample).
    fs : float
        Sampling frequency in Hz.
    T : float
        Duration in seconds; the sample count is round(T * fs).
    phi : float, optional
        Initial phase in radians.

    Returns
    -------
    numpy.ndarray of float64
        The chirp samples, length N = round(T * fs).
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

def square(f0, fs, N, phi=0):
    """
    Band-limited square wave: odd harmonics with k*f0 < fs/2 only.

    Parameters
    ----------
    f0 : float
        Frequency of square wave in Hz (at sample 0).
    fs : float
        Sampling frequency in Hz.
    N : int
        Number of samples in square wave.
    phi : float, optional
        Initial phase in radians.

    Returns
    -------
    numpy.ndarray of float64
        The square wave samples, length N.
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

def sawtooth(f0, fs, N, phi=0.0):
    """
    Band-limited rising sawtooth: all harmonics with k*f0 < fs/2.

    Zero-crossing at n = 0 for phi = 0; use phi = pi to match
    scipy.signal.sawtooth's convention (starts at -1).

    Parameters
    ----------
    f0 : float
        Fundamental frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    N : int
        Number of samples.
    phi : float, optional
        Initial phase in radians.

    Returns
    -------
    numpy.ndarray of float64
        The sawtooth samples, length N.
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