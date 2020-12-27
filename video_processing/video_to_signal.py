import numpy as np


def get_avg_color_signal(frames, channel=1):
    return np.array([frame[:, :, channel].mean() for frame in frames])
