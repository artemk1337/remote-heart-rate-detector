from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2


def show_image(img):
    Image.fromarray(img).show()


def image_normalization(img):
    img -= np.min(img)
    img /= np.max(img)
    img *= 255
    img = img.astype(np.uint8)
    return img

def visualize_f(faces_dict, mean_signals_norm, fft, areas, fps, buff_size, filename='tmp', speedx=0.25):
    for area in areas:
        frames_to_save = []
        keys = [int(key) for key in mean_signals_norm.keys()]
        fig = plt.figure(figsize=(5, 10), dpi=100)
        for key in tqdm(keys):
            plt.subplot(3, 1, 1)
            plt.title(f"Кадр: {key}, размер буфера: {buff_size}")
            plt.imshow(faces_dict[key]['Rectangled'].copy())
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
        save_video(frames_to_save, filename, area, fps, 0.2)


def save_video(all_imgs, filename, area, fps, save_speedx):
    all_imgs = np.array(all_imgs).astype(np.uint8)
    out = cv2.VideoWriter(filename + area + '_pulse.avi',
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          int(fps * save_speedx),
                          (all_imgs[0].shape[1], all_imgs[0].shape[0]))
    print("========== Save File ==========")
    for im in tqdm(all_imgs):
        out.write(im)
    out.release()
