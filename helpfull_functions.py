from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import os


def show_image(img):
    Image.fromarray(img).show()


def image_normalization(img):
    img -= np.min(img)
    img /= np.max(img)
    img *= 255
    img = img.astype(np.uint8)
    return img

def visualize_f(faces_dict, mean_signals_norm, fft, areas, fps, buff_size, channel, step, filename='tmp', speedx=0.25):
    for area in areas:
        frames_to_save = []
        keys = [int(key) for key in mean_signals_norm.keys()]
        keys_preview = np.array([int(key) for key in faces_dict.keys()])
        fig = plt.figure(figsize=(5, 10), dpi=100)
        for key in tqdm(keys):
            plt.subplot(3, 1, 1)
            key_ = (keys_preview[keys_preview < key]).max()
            plt.title(f"Кадр: {key_}, размер буфера: {buff_size}, канал: {channel}, шаг: {step}")
            im_cv = faces_dict[key_]['Rectangled'].copy()
            plt.imshow(cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB))
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 1, 2)
            plt.plot(mean_signals_norm[key][area])
            plt.legend(['Signal'], loc=1)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(3, 1, 3)
            plt.plot(fft[key][area][0], fft[key][area][1])
            plt.legend(['Fourier'], loc=1)
            canvas = FigureCanvasAgg(fig)
            s, (width, height) = canvas.print_to_buffer()
            X = np.fromstring(s, np.uint8).reshape((height, width, 4))
            # display(fig)
            plt.clf()
            # clear_output(wait=True)
            X = cv2.cvtColor(X, cv2.COLOR_RGBA2BGR)
            frames_to_save += [X[80:-80]]
        save_video(frames_to_save, filename, area, fps, speedx)


def save_video(all_imgs, filename, area, fps, save_speedx):
    all_imgs = np.array(all_imgs).astype(np.uint8)
    add_postfix = 1
    full_save_filename = filename + area + '_pulse.avi'

    if os.path.isfile(full_save_filename):
        full_save_filename = filename + area + '_pulse_' + str(add_postfix) + '.avi'
        while os.path.isfile(full_save_filename):
            add_postfix += 1
            full_save_filename = filename + area + '_pulse_' + str(add_postfix) + '.avi'

    out = cv2.VideoWriter(full_save_filename,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          max(1, round(fps * save_speedx)),
                          (all_imgs[0].shape[1], all_imgs[0].shape[0]))
    print("========== Save File ==========")
    print(full_save_filename)
    for im in tqdm(all_imgs):
        out.write(im)
    out.release()
