import seaborn as sns

color_palette = 'viridis'

sns.set_theme(palette=color_palette)

figure_dimensions = dict(square=(6, 6), wrect=(8, 4), hrect=(4, 8), refs=(6, 8))

label_font = dict(size=14, name='Courier', family='serif', style='italic', weight='bold')
suptitle_font = dict(size=20, name='Courier', family='serif', style='normal', weight='bold')
title_font = dict(size=16, name='Courier', family='sans', style='italic', weight='normal')


def interpolator(x, y, z):
    import numpy as np
    from scipy.interpolate import interp2d

    f = interp2d(x, y, z, kind='cubic')

    x1 = np.linspace(min(x), max(x), int(x.size * 1.5))
    y1 = np.linspace(min(y), max(y), int(y.size * 1.5))
    z1 = f(x1, y1)

    x2, y2 = np.meshgrid(x1, y1)

    return x2, y2, z1


def save_and_close(plt, fig, filepath):
    from pathlib import Path
    import os

    path = Path(filepath)
    if not os.path.isdir(path.parents[0]):
        os.mkdir(path.parents[0])
    plt.savefig(filepath)
    fig.clear()
    plt.close()
    plt.cla()
    plt.clf()


def basic_figure(shape, **kwargs):
    import matplotlib.pyplot as plt

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions[shape])

    ax = fig.add_subplot()

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    return plt, fig, ax


def polar_figure(shape, **kwargs):
    import matplotlib.pyplot as plt

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions[shape])

    ax = fig.add_subplot(projection='polar')

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(90)

    ax.tick_params(labelsize=20)

    ax.annotate('', xy=(0.5, 1), xycoords='axes fraction',
                arrowprops=dict(facecolor='red', width=6, headwidth=12, alpha=0.75))
    ax.annotate('Face forward', xy=(0.5, 0.75), xytext=(0.5, 0.75), xycoords='axes fraction', ha='center', va='center',
                fontsize=20, rotation=90)
    return plt, fig, ax


def set_parameters(ax, **kwargs):
    if 'xlim' in kwargs.keys():
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs.keys():
        ax.set_ylim(kwargs['ylim'])
    if 'zlim' in kwargs.keys():
        ax.set_zlim(kwargs['zlim'])
    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'], **label_font)
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'], **label_font)
    if 'zlabel' in kwargs.keys():
        ax.set_zlabel(kwargs['zlabel'], **label_font)


def wave(x, filepath=None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np

    num_frames, num_channels = x.size, x.ndim

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['wrect'])

    ax = fig.subplots(num_channels, 1)

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)

    if num_channels == 1:
        ax = [ax]
        x = x.reshape((-1, 1))

    for c in range(num_channels):
        ax[c].grid(True)

        if 't' in kwargs.keys():
            t = kwargs['t'][0]
        elif 'fs' in kwargs.keys():
            fs = kwargs['fs'][0]
            t = np.arange(0, num_frames) / fs
        else:
            t = np.linspace(0, 1, num_frames)
        if 'xlim' in kwargs.keys():
            ax[c].set_xlim(kwargs['xlim'][c])
        if 'ylim' in kwargs.keys():
            ax[c].set_ylim(kwargs['ylim'][c])
        if 'xlabel' in kwargs.keys():
            ax[c].set_xlabel(kwargs['xlabel'][c])
        if 'ylabel' in kwargs.keys():
            ax[c].set_ylabel(kwargs['ylabel'][c])
        if 'title' in kwargs.keys():
            ax[c].set_title(kwargs['title'][c])
        ax[c].plot(t, x[:, c])

    if filepath is not None:
        save_and_close(plt, fig, filepath)
        return filepath

    return plt


def waves(x, filepath=None, **kwargs):
    import numpy as np

    num_frames, num_channels = x.size, x.ndim

    plt, fig, ax = basic_figure('wrect', **kwargs)

    if 't' in kwargs.keys():
        t = kwargs['t']
    elif 'fs' in kwargs.keys():
        fs = kwargs['fs'][0]
        t = np.arange(0, num_frames) / fs
    else:
        t = np.linspace(0, 1, num_frames)

    set_parameters(ax, **kwargs)

    handles = []
    for i in range(num_channels):
        if 'legend_labels' in kwargs.keys() and len(kwargs['legend_labels']) <= num_channels:
            label = kwargs['legend_labels'][i]
        else:
            label = ''
        handle, = ax.plot(t, x[:, i], label=label)
        handles.append(handle)

    if 'legend' in kwargs.keys() and kwargs['legend']:
        ax.legend(handles=handles)

    if filepath is not None:
        save_and_close(plt, fig, filepath)
        return filepath

    return plt


def spectrogram(x, filepath=None, **kwargs):
    import matplotlib.pyplot as plt

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['square'])

    ax = fig.add_subplot()

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)
    if 'fs' in kwargs.keys():
        ax.specgram(x, cmap=color_palette, Fs=kwargs['fs'])
    else:
        ax.specgram(x, cmap=color_palette)

    if filepath is not None:
        save_and_close(plt, fig, filepath)
        return filepath

    return plt


def spectrum(x, fs, spectrum_type='amplitude', filepath=None, **kwargs):
    import numpy as np
    from numpy.fft import fftfreq
    from transforms import amplitude_spectrum, power_spectrum, phase_spectrum, log_spectrum

    if spectrum_type == 'amplitude':
        s = amplitude_spectrum(x)
    elif spectrum_type == 'power':
        s = power_spectrum(x)
    elif spectrum_type == 'phase':
        s = phase_spectrum(x)
    elif spectrum_type == 'log':
        s = log_spectrum(x)
    else:
        s = x

    step = 1 / fs
    freqs = fftfreq(x.size, step)
    idx = np.argsort(freqs)

    n = int(np.ceil(x.size / 2))

    s = s[idx][n:]
    freqs = freqs[idx][n:]

    plt, fig, ax = basic_figure('wrect', **kwargs)
    ax.plot(freqs, s)

    if filepath is not None:
        save_and_close(plt, fig, filepath)

    return plt


def cepstrum(x, fs, offset, window_length, filepath=None, **kwargs):
    from transforms import cepstrum

    c, q = cepstrum(x, fs, offset, window_length)

    plt, fig, ax = basic_figure('wrect', **kwargs)
    ax.plot(q * 1000, c)

    if filepath is not None:
        save_and_close(plt, fig, filepath)

    return plt


def hrtfs(zeniths, azimuths, filepath=None, **kwargs):
    import numpy as np

    plt, fig, ax = polar_figure('square', **kwargs)

    ax.set_rmax(90)
    ax.set_rticks(np.linspace(-30, 75, 8))
    ax.set_rlim(bottom=90, top=-40)

    ax.scatter(np.radians(azimuths), zeniths)

    if filepath is not None:
        save_and_close(plt, fig, filepath)
        return filepath

    return plt


def reflections(zeniths, azimuths, delays, amplitudes, delay_max=None, filepath=None, **kwargs):
    import numpy as np
    from data_loader import list_hrtf_data

    zmin = min(list_hrtf_data().keys())
    zmax = max(list_hrtf_data().keys())
    if delay_max is None:
        delay_max = max(delays)

    plt, fig, ax = polar_figure('refs', **kwargs)

    theta = np.radians(azimuths)
    r = np.array(delays)
    area = ((np.array(zeniths) - zmin) / (zmax - zmin)) * (1500 - 200) + 200
    color = np.array(amplitudes)

    norm = plt.Normalize(0, 1)
    ax.set_rmax(delay_max)

    ax.text(np.radians(135), delay_max + 15, 'Azimuth', ha='center', va='center', fontsize=20, rotation=45)
    ax.text(np.radians(95), delay_max / 2., 'Time/delay, ms', ha='center', va='center', fontsize=20)

    h1 = ax.scatter([], [], s=150, c='k', alpha=0.5)
    h2 = ax.scatter([], [], s=600, c='k', alpha=0.5)
    h3 = ax.scatter([], [], s=1050, c='k', alpha=0.5)
    h4 = ax.scatter([], [], s=1500, c='k', alpha=0.5)

    handles = (h1, h2, h3, h4)
    labels = ('-45', '0', '45', '90')

    ax.legend(handles, labels, scatterpoints=1, loc='upper left', title='Zenith', title_fontsize=20, frameon=True,
              fancybox=True, fontsize=16, bbox_to_anchor=(0., -0.2, 1., .15), ncol=4, mode='expand', borderaxespad=.1,
              borderpad=1)

    ax.scatter(theta, r, c=color, s=area, cmap=color_palette, norm=norm, alpha=0.9)

    sns.set_style('darkgrid', {'axes.grid': False})

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_palette), ax=ax, orientation='horizontal')
    cb.set_label('Amplitude', size=20)
    cb.ax.tick_params(labelsize=16)

    if filepath is not None:
        save_and_close(plt, fig, filepath)

    return plt, fig


def sample(zenith, azimuth, zeniths, azimuths, amplitudes, delays, amplitude, delay, time, delay_max=None,
           time_max=None, filepath=None, **kwargs):
    import numpy as np
    from data_loader import list_hrtf_data

    if delay_max is None:
        delay_max = max(delays)
    if time_max is None:
        time_max = time

    zmin = min(list_hrtf_data().keys())
    zmax = max(list_hrtf_data().keys())

    area = ((zenith - zmin) / (zmax - zmin)) * (0.5 - 0.1) + 0.1
    err = (time / time_max) * (0.5 - 0.1) + 0.1

    plt, fig = reflections(zeniths, azimuths, delays, amplitudes, **kwargs)

    cmap = plt.get_cmap(color_palette)

    ax = plt.gca()
    ax.bar(np.deg2rad(azimuth), delay_max - delay, xerr=err, bottom=delay, width=area, alpha=0.5, color=cmap(amplitude),
           ecolor=cmap(amplitude))

    if filepath is not None:
        save_and_close(plt, fig, filepath)

    return plt


def binaural_activity_map_2d(z, filepath=None, **kwargs):
    import numpy as np

    z = z.transpose()
    x = np.linspace(-1, 1, z.shape[1])
    y = np.linspace(0, 50, z.shape[0])

    x, y, z = interpolator(x, y, z)
    plt, fig, ax = basic_figure('square', **kwargs)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 50)

    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'], **label_font)
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'], **label_font)

    surf = ax.contourf(x, y, z, cmap=color_palette)

    sns.set_style("darkgrid", {'axes.grid': False})
    cb = plt.colorbar(surf, ax=ax)
    cb.set_label('Correlation', size=20)
    cb.ax.tick_params(labelsize=16)

    if filepath is not None:
        save_and_close(plt, fig, filepath)

    return plt


def binaural_activity_map_3d(z, filepath=None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np

    z = z.transpose()
    x = np.linspace(-1, 1, z.shape[1])
    y = np.linspace(0, 50, z.shape[0])

    x, y, z = interpolator(x, y, z)

    fig = plt.figure(layout='constrained')
    fig.set_size_inches(figure_dimensions['square'])

    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 50)
    ax.view_init(elev=80)

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    set_parameters(ax, **kwargs)

    sns.set_style('darkgrid', {'axes.grid': False})
    surf = ax.plot_surface(x, y, z, cmap=color_palette, rstride=1, cstride=1, edgecolor='none', aa=False)

    cb = plt.colorbar(surf, ax=ax)
    cb.set_label('Correlation', size=20)
    cb.ax.tick_params(labelsize=16)

    if filepath is not None:
        save_and_close(plt, fig, filepath)

    return plt


def zenith_range(zmin, zmax, filepath=None, **kwargs):
    import matplotlib as mpl
    import numpy as np

    cmap = mpl.cm.get_cmap(color_palette)(0.5)

    zmin = np.deg2rad(zmin)
    zmax = np.deg2rad(zmax)
    theta = np.linspace(zmin, zmax)

    plt, fig, ax = polar_figure('square', **kwargs)

    if 'suptitle' in kwargs.keys():
        plt.suptitle(kwargs['suptitle'], **suptitle_font)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], **title_font)

    ax.set_rticks([])
    ax.tick_params(labelsize=20)

    ax.annotate('', xy=(1, 0.5), xytext=(0.5, 0.5), xycoords='axes fraction',
                arrowprops=dict(facecolor='red', width=6, headwidth=12, alpha=0.75))

    ax.annotate('Face forward', xy=(0.5, 0.5), xytext=(0.75, 0.5), xycoords='axes fraction', ha='center', va='center',
                fontsize=20)
    ax.grid(True)
    area = plt.fill_between(theta, 0, 1, alpha=0.75, label='area', color=cmap)

    if filepath is not None:
        save_and_close(plt, fig, filepath)
        return filepath

    return plt, fig, ax, area


def azimuth_range(amin, amax, filepath=None, **kwargs):
    import matplotlib as mpl
    import numpy as np

    cmap = mpl.cm.get_cmap(color_palette)(0.5)

    amin = np.deg2rad(amin)
    amax = np.deg2rad(amax)
    theta = np.linspace(amin, amax)

    plt, fig, ax = polar_figure('square', **kwargs)
    ax.set_rticks([])
    area = plt.fill_between(theta, 0, 1, alpha=0.75, label='area', color=cmap)

    if filepath is not None:
        save_and_close(plt, fig, filepath)
        return filepath

    return plt, fig, ax, area


def mfcc(x, filepath=None, **kwargs):
    from transforms import mfcc
    left, right = mfcc(x)

    return spectrogram(left, filepath=filepath, **kwargs)


def autocorrelation(x, filepath=None, **kwargs):
    pass


def second_layer_autocorrelation(x, filepath=None, **kwargs):
    pass


def background():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    deflection = 30
    damping_coefficient = 0.75
    w = 2 * np.pi

    x = np.linspace(0, 2 * np.pi, 257)
    y = np.cos(x)

    el = deflection * np.exp(-y * damping_coefficient) * np.cos(w * y)

    n = 20
    cmap = mpl.cm.get_cmap('twilight_shifted')(np.linspace(0, 1, n))

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(12, 8)

    ax.set_facecolor('k')
    ax.grid(False)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    for i in range(n):
        plt.plot(x, np.log(i + 1) * el, color=cmap[i], alpha=0.75)

    plt.show()
