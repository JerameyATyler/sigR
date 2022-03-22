def apply_hrtf(x, zenith, azimuth):
    from utils import arrays_to_audiosegment, split_channels
    from pydub import AudioSegment
    from data_loader import list_hrtf_data
    from scipy.signal import convolve

    left, right = sorted(list(list_hrtf_data()[zenith][azimuth]))
    left = AudioSegment.from_wav(left)
    right = AudioSegment.from_wav(right)
    h = AudioSegment.from_mono_audiosegments(left, right)

    fs = max(h.frame_rate, x.frame_rate)

    if fs > h.frame_rate:
        h = h.set_frame_rate(fs)
    if fs > x.frame_rate:
        x = x.set_frame_rate(fs)

    left_x, right_x = split_channels(x)
    left_h, right_h = split_channels(h)

    left = convolve(left_h, left_x)
    right = convolve(right_h, right_x)

    return arrays_to_audiosegment(left, right, fs)


def apply_reflection(x, amplitude, delay, zenith, azimuth):
    from pydub import AudioSegment
    from utils import arrays_to_audiosegment, split_channels

    x = apply_hrtf(x, zenith, azimuth)
    fs = x.frame_rate

    left, right = split_channels(x)
    left *= amplitude
    right *= amplitude
    y = AudioSegment.silent(delay) + arrays_to_audiosegment(left, right, fs)

    return y


def apply_reverberation(x, amplitude, delay, time):
    from pydub import AudioSegment
    import numpy as np
    from scipy.signal import fftconvolve
    from utils import split_channels, arrays_to_audiosegment

    fs = x.frame_rate

    length = fs * time * 2
    t = np.linspace(0, int(np.ceil(length / fs)), int(length + 1))
    envelope = np.exp(-1 * (t / time) * (60 / 20) * np.log(10)).transpose()

    left_reverb = np.random.randn(t.shape[0], ) * envelope
    right_reverb = np.random.randn(t.shape[0], ) * envelope

    left, right = split_channels(x)

    left = fftconvolve(left_reverb, left) * amplitude
    right = fftconvolve(right_reverb, right) * amplitude
    y = AudioSegment.silent(delay) + arrays_to_audiosegment(left, right, fs)

    return y


def mix_parts(parts, offset, duration):
    from pydub import AudioSegment

    if duration == -1:
        sounds = [AudioSegment.from_mp3(p) for p in parts]
    else:
        sounds = [AudioSegment.from_mp3(p)[offset:offset + duration * 1000] for p in parts]
    t = max([len(p) for p in sounds])

    s = AudioSegment.silent(duration=t)

    for p in sounds:
        s = s.overlay(p, gain_during_overlay=-3)

    return s


def mix_reflections(x, count, amplitudes, delays, zeniths, azimuths):
    from pydub import AudioSegment

    reflections = [apply_reflection(x, amplitudes[r], delays[r], zeniths[r], azimuths[r]) for r in range(count)]
    t = max([len(r) for r in reflections])

    s = AudioSegment.silent(duration=t)
    y = s.overlay(x, gain_during_overlay=-3)

    for r in reflections:
        y = y.overlay(r, gain_during_overlay=-3)

    return y


def sum_signals(x, y):
    from pydub import AudioSegment

    t = max(len(x), len(y))

    s = AudioSegment.silent(duration=t)
    s = s.overlay(x, gain_during_overlay=-3)
    s = s.overlay(y, gain_during_overlay=-3)

    return s


def adjust_signal_to_noise(x, dB):
    from pydub.generators import WhiteNoise

    noise = WhiteNoise().to_audio_segment(duration=len(x))
    return noise.overlay(x, gain_during_overlay=dB)
