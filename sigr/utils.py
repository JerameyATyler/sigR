def audiosegment_to_array(audiosegment):
    import numpy as np

    y = np.array(audiosegment.get_array_of_samples())
    if audiosegment.channels == 2:
        y = y.reshape((-1, 2))

    return np.float32(y) / 2 ** 15


def array_to_audiosegment(arr, fs):
    from pydub import AudioSegment
    import numpy as np

    y = np.int16(arr * 2 ** 15)
    return AudioSegment(y.tobytes(), frame_rate=fs, sample_width=2, channels=1)


def arrays_to_audiosegment(left, right, fs):
    from pydub import AudioSegment

    left = array_to_audiosegment(left, fs)
    right = array_to_audiosegment(right, fs)
    return AudioSegment.from_mono_audiosegments(left, right)


def split_channels(x):
    if x.channels == 1:
        x = x.set_channels(2)

    channels = x.split_to_mono()
    left = audiosegment_to_array(channels[0])
    right = audiosegment_to_array(channels[1])

    return left, right


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False

        if shell == 'google.colab.__shell':
            return True
        return False
    except NameError:
        return False


def play_audio(x):
    from IPython.display import Audio
    import numpy as np

    fs = x.frame_rate

    left, right = split_channels(x)

    y = left

    if x.channels == 2:
        y = np.array([left, right])
    return Audio(y, rate=fs)


def generate_impulse(duration):
    import numpy as np

    fs = 48000
    click = np.ones(1)
    click = np.concatenate((click, np.zeros(fs * duration - 1)))
    return array_to_audiosegment(click, fs)
