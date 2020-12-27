from scipy import signal
import numpy as np
import time

from video_processing.color_magnification import magnify_color
from video_processing.video_to_signal import get_avg_color_signal


def fourier_for_norm_signal(mean_signal_norm, fps):
    L = len(mean_signal_norm)

    raw = np.fft.rfft(mean_signal_norm * 30)  # do real fft with the normalization multiplied by 10
    freqs_ = float(fps) / L * np.arange(L // 2 + 1)
    freqs = 60. * freqs_

    fft = np.abs(raw) ** 2  # get amplitude spectrum
    # plot(freqs, fft, 'freq', 'Fourier before pruning')

    idx = np.where((freqs > 50) & (freqs < 180))  # the range of frequency that HR is supposed to be within
    pruned = fft[idx]
    pfreq = freqs[idx]

    freqs_ = pfreq
    fft = pruned
    # plot(pfreq, fft, 'freq', 'Fourierr after pruning')
    fft = [pfreq, fft]
    idx = np.argmax(pruned)  # max in the range can be HR
    # print(fft[0][idx])
    heart_rate = fft[0][idx]
    return fft, heart_rate


def calculate_pulse_in_buffer(buffer, L, fps, channel, levels):
    ## Обработка кадров видео ##
    magnified_ROI = magnify_color(buffer, fps, low=0.4, high=2, levels=levels, amplification=50)
    # print(magnified_ROI.shape)
    mean_signal = get_avg_color_signal(magnified_ROI, channel)
    # time.sleep(0.02)
    # print(mean_signal.shape)

    ## Делаем сигнал более переодичным ##
    # Проблемы, что память не успевает выделяться
    try:
        detrend = signal.detrend(mean_signal)
    except:
        time.sleep(0.1)
        detrend = signal.detrend(mean_signal)
    time_range = np.array([1 / fps * i for i in range(L)])
    interpolated = np.interp(time_range, time_range, detrend)  # interpolation by 1
    interpolated = np.hamming(L) * interpolated  # make the signal become more periodic
    mean_signal_norm = interpolated / np.linalg.norm(interpolated)

    ## Фурье ##
    fft, heart_rate = fourier_for_norm_signal(mean_signal_norm, fps)
    return mean_signal, mean_signal_norm, fft, heart_rate


def algorithm(faces_dict, fps, idx_start=1, idx_end=1000, area='FullFace', channel=1, levels=3):
    idx_array = np.array([key for key in faces_dict.keys()]).astype(str).astype(int)
    idx_array = idx_array[(idx_array < idx_end) & (idx_array >= idx_start)]

    L = len(idx_array)

    buffer = np.array([faces_dict[i][area].copy() for i in idx_array])
    # print(buffer.shape)

    mean_signal, mean_signal_norm, fft, heart_rate = calculate_pulse_in_buffer(buffer, L, fps, channel, levels)

    return mean_signal, mean_signal_norm, fft, heart_rate