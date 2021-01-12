from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import medfilt, spline_filter
from torch import device as device_
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from scipy import signal
from torch import cuda
from PIL import Image
from tqdm import tqdm
import configparser
import numpy as np
import warnings
import types
import time
import sys
import cv2

from video_processing.extract_face_from_video import extract_frame_from_video
from video_processing.preprocessing_frames import fill_None, resize_images_to_one_shape
from video_processing.create_ROI import create_ROI
from video_processing.main_algorithm import algorithm
from helpfull_functions import visualize_f
from signal_processing.FastICA import apply_FastICA


warnings.filterwarnings("ignore")
device = device_('cuda:0' if cuda.is_available() else 'cpu')


def main(
    visualize,
    FastICA,
    areas,
    areas_for_ICA,
    idx_start, idx_end,
    step,
    buff_size,
    channel,
    n_components,
    levels,
    speedx,
    filename_video,
    filename_ICA
):
    """

    :param visualize:
    :param FastICA:
    :param areas:
    :param areas_for_ICA:
    :param idx_start:
    :param idx_end:
    :param step:
    :param buff_size:
    :param channel:
    :param n_components:
    :param levels:
    :param speedx:
    :param filename_video:
    :param filename_ICA:

    :return:
    """

    faces_dict, fps, length = extract_frame_from_video(sys.argv[-1], idx_start, idx_end, step=step)
    fill_None(faces_dict)
    resize_images_to_one_shape(faces_dict)
    create_ROI(faces_dict)

    mean_signals_norm = {}
    fft = {}
    heart_rates = {}

    print('<===== Process signal =====>')
    for i in tqdm(range(buff_size + 1, length, step)):
        mean_signals_norm[i] = {}
        fft[i] = {}
        heart_rates[i] = {}
        for area in areas:
            (mean_signal, mean_signals_norm[i][area],
             fft[i][area], heart_rates[i][area]) = algorithm(faces_dict, fps, i - buff_size, i, area, channel, levels)

    if visualize is True:
        print('<===== Visualize =====>')
        visualize_f(faces_dict, mean_signals_norm, fft, areas, fps, buff_size,
                    filename=filename_video, speedx=speedx)

    if FastICA is True:
        print('<===== Apply FastICA =====>')
        mean_signals = []
        for area in areas_for_ICA:
            mean_signals = [mean_signals_norm[key][area] for key in mean_signals_norm.keys()]
            apply_FastICA(mean_signals, fps, n_components, filename=filename_ICA + area)


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('settings.cfg')

    visualize = eval(config.get("default", "visualize"))
    FastICA = eval(config.get("default", "FastICA"))
    areas = eval(config.get("areas", "areas"))
    areas_for_ICA = eval(config.get("areas", "areas_for_ICA"))
    idx_start, idx_end = int(config.get("video", "idx_start")), eval(config.get("video", "idx_end"))
    step = int(config.get("video", "step"))
    buff_size = int(config.get("signal", "buff_size"))
    channel = int(config.get("signal", "channel"))
    n_components = int(config.get("signal", "n_components"))
    levels = int(config.get("color magnification", "levels"))

    speedx = float(config.get("save", "speedx"))
    filename_video = eval(config.get("save", "filename_video"))
    filename_ICA = eval(config.get("save", "filename_ICA"))

    assert len(sys.argv) == 2, "Args should be 2: python <program name> <filename>"
    main(visualize,
    FastICA,
    areas,
    areas_for_ICA,
    idx_start, idx_end,
    step,
    buff_size,
    channel,
    n_components,
    levels,
    speedx,
    filename_video,
    filename_ICA)
