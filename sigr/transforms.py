def cross_correlation(x, y, lag):
    from scipy.signal import correlate

    nx = len(x)
    if nx != len(y):
        raise ValueError('x and y must have the same length')

    c = correlate(x, y)

    if lag is None:
        lag = nx - 1

    if lag >= nx or lag < 1:
        raise ValueError

    return c[nx - 1 - lag: nx + lag]


def normalized_cross_correlation(x, y, lag):
    import numpy as np

    c = cross_correlation(x, y, lag)

    s = np.sqrt(sum(x ** 2) * sum(y ** 2))

    if s == 0:
        c *= 0
    else:
        c /= s
    return c


def autocorrelation(x):
    from utils import split_channels

    lag = (1 * int(x.frame_rate / 1000))
    left, right = split_channels(x)
    xc = cross_correlation(left, right, lag)
    xc = xc.reshape(-1, 1)

    return xc


def second_layer_autocorrelation(x):
    from utils import array_to_audiosegment

    y = autocorrelation(x)
    y = array_to_audiosegment(y, x.frame_rate)
    return autocorrelation(y)


def amplitude_spectrum(x):
    from numpy.fft import fft
    import numpy as np
    return np.abs(fft(x))


def power_spectrum(x):
    return amplitude_spectrum(x) ** 2


def phase_spectrum(x):
    from numpy.fft import fft
    import numpy as np
    return np.angle(fft(x))


def log_spectrum(x):
    import numpy as np
    return np.log(amplitude_spectrum(x))


def cepstrum(x, fs, offset, window_length):
    from scipy.signal.windows import hamming
    from numpy.fft import ifft
    import numpy as np

    w = hamming(window_length, False)

    x = x[offset:offset + window_length] * w

    number_unique_points = int(np.ceil((window_length + 1) / 2))

    c = np.real(ifft(log_spectrum(x)))[0:number_unique_points]
    q = np.arange(0, number_unique_points) / fs

    return c, q


def cepstral_autocorrelation(x):
    from utils import split_channels, arrays_to_audiosegment

    offset = 1024
    window_length = offset * 64 * 2
    fs = x.frame_rate

    left, right = split_channels(x)
    cl, _ = cepstrum(left, fs, offset, window_length)
    cr, _ = cepstrum(right, fs, offset, window_length)

    return autocorrelation(arrays_to_audiosegment(cl, cr, fs))


def cepstral_second_layer_autocorrelation(x):
    from utils import array_to_audiosegment

    y = cepstral_autocorrelation(x)
    print(y.shape)
    return x


def mfcc(x):
    import librosa
    import numpy as np
    from utils import split_channels

    left, right = split_channels(x)
    fs = x.frame_rate

    mfcc_l = librosa.feature.mfcc(y=left, sr=fs, n_mfcc=256)
    mfcc_r = librosa.feature.mfcc(y=right, sr=fs, n_mfcc=256)

    return np.stack([mfcc_l, mfcc_r])
