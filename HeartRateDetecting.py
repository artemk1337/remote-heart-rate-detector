import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import types
from tqdm import tqdm
from scipy.signal import medfilt, spline_filter
import scipy.fftpack as fftpack
from scipy import signal
from facenet_pytorch import MTCNN
from torch import device as device_
from torch import cuda
import warnings
warnings.filterwarnings("ignore")


device = device_('cuda:0' if cuda.is_available() else 'cpu')


##### Видео-фильтры #####

def gaussian_video(video_tensor,levels=3):
    def build_gaussian_pyramid(src,level=3):
        s=src.copy()
        pyramid=[s]
        for i in range(level):
            s=cv2.pyrDown(s)
            pyramid.append(s)
        return pyramid

    for i in range(0,len(video_tensor)):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((len(video_tensor),len(gaussian_frame),gaussian_frame[0].shape[0],3))
        vid_data[i]=gaussian_frame
    return vid_data

def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

def amplify_video(gaussian_vid,amplification=70):
    return gaussian_vid*amplification

def reconstract_video(amp_video,origin_video,levels=3):
    final_video=np.zeros(np.shape(origin_video))
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img=cv2.resize(img, (origin_video[i].shape[0], origin_video[i].shape[1]))
        img=img+origin_video[i]
        final_video[i]=img
    return final_video.astype(np.float16)

def add_filters_video(data_buffer,fps,low=0.4,high=2,levels=3,amplification=30):
    gau_video=gaussian_video(data_buffer,levels=levels)
    filtered_tensor=temporal_ideal_filter(gau_video,low,high,fps)
    amplified_video=amplify_video(filtered_tensor,amplification=amplification)
    final_video = reconstract_video(amplified_video,data_buffer,levels=levels)
    return final_video

##### Видео-фильтры #####


def image_normalization(img):
    img -= np.min(img)
    img /= np.max(img)
    img *= 255
    img = img.astype(np.uint8)
    return img


def get_avg_color_signal(frames):
    mean = [frame[:, :, 1].mean() for frame in frames]
    return np.array(mean)


class PulseAnalyzer:
    def __init__(self, buff_size=100, save_speedx=1., filename=None, visualize=True):
        self.box_shift = None
        self.buff_size = buff_size
        self.save_speedx = save_speedx
        self.time_full = None
        self.fps = None
        self.visualize = visualize

        self.frames = []
        self.frame_ids = []
        self.centers = []
        self.faces = []

        self.ROI_coords = {}
        self.frames_ROI = {}

        # buffers
        self.frames_filtered = {}
        self.mean_g_signals = {}
        self.mean_g_signals_filtered = {}
        self.mean_g_signals_norm = {}
        self.fft = {}

        self.pulse_buff = []
        # self.times_s = []

        self.L = None
        self.time_s = None

        if filename:
            self.filename = filename
            self.get_frames_from_videofile(filename)

    # get all frames from video file
    def get_frames_from_videofile(self, filename):
        self.filename = filename
        self.frames = []
        self.frame_ids = []
        
        cap = cv2.VideoCapture(filename)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        i = 0
        while 1:
            ret, img = cap.read()
            if not ret:
                break
            self.frames += [img]
            self.frame_ids += [i]
            i += 1

    def find_face(self):
        """
        find face on the frames

        create:
            self.faces
            self.frame_ids
        """
        def del_skipped_frames():
            idxs = [idx for idx, val in enumerate(self.centers) if val == 0]
            for index in sorted(idxs, reverse=True):
                del self.centers[index]
                del self.frames[index]
                del self.frame_ids[index]
        
        def medfilt_filter(step=7):
            y_ = medfilt([i[0] for i in self.centers], step)
            x_ = medfilt([i[1] for i in self.centers], step)
            return y_, x_
            

        self.centers, h_shift, w_shift, centers = ([], [], [], None)

        # fast mtcnn pytorch; uses with cuda
        if cuda.is_available():
            frames_cropped = []
            box_prev = None
            
            mtcnn = MTCNN(image_size=200, device=device)
            for frame in tqdm(self.frames):
                box, _ = mtcnn.detect(frame)
                if box is not None:
                    box = np.array(box[0]).astype(int)
                    x1, x2, y1, y2 = box[1], box[3], box[0], box[2]
                    h_shift += [(y2 - y1) // 2]
                    w_shift += [(x2 - x1) // 2]
                    centers = [y1 + h_shift[-1], x1 + w_shift[-1]]
                    #plt.imshow(frame[x1:x2, y1:y2])
                    #plt.show()
                    if centers is not None:
                        self.centers += [centers]
                    else:
                        self.centers += [0]
                else:
                    self.centers += [0]

            del mtcnn
            
            del_skipped_frames()

        # haard; uses without cuda
        else:
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
            for frame in tqdm(self.frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray)
                for (x, y, w, h) in faces:
                    h_shift += [h // 2]
                    w_shift += [w // 2]
                    centers = [y + h // 2, x + w // 2] 
                if centers is not None:
                    self.centers += [centers]
                else:
                    self.centers += [0]
            
            del face_cascade

            del_skipped_frames()

        self.box_shift = [np.mean(w_shift, dtype=int), np.mean(h_shift, dtype=int)]
        # drop discharges from signal
        if len(self.centers) == 0: raise ValueError("Невозможно определить лицо")
        y_, x_ = medfilt_filter()

        self.centers = [[int(y), int(x)] for x, y in zip(x_, y_)]
        for frame, (y, x) in tqdm(zip(self.frames, self.centers)):
            face = frame[x - self.box_shift[0]:x + self.box_shift[0],
                         y - self.box_shift[1]:y + self.box_shift[1]]
            self.faces += [face]


    def add_rectangles_img(self, img) -> np:
        def rectangle(img, x, y, w, h):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        img = np.array(img, copy=True)
        if self.frames_ROI.get('forehead'):
            rectangle(img, *self.ROI_coords['forehead'])
        if self.frames_ROI.get('left_cheek'):
            rectangle(img, *self.ROI_coords['left_cheek'])
        if self.frames_ROI.get('right_cheek'):
            rectangle(img, *self.ROI_coords['right_cheek'])
        return img

    def show_rectangles(self, frames=None, show_ROI=False):
        def show_imgs(imgs):
            for i in range(len(imgs)):
                img = self.add_rectangles_img(imgs[i])
                cv2.imshow('face', img)
                k = cv2.waitKey(int(self.fps)) & 0xff
                if k == 27:
                    break

        if frames is None: frames = self.faces
        show_imgs(frames)
        cv2.destroyAllWindows()


    def __create_ROI(self):
        def create_ROI_(frames, x=None, y=None, w=None, h=None):
            tmp = []
            for frame in frames:
                if x and y and w and h:
                    img = np.array(frame[y:y + h, x:x + w], copy=True)
                else:
                    img = cv2.resize(np.array(frame, copy=True), (max(frame.shape[0], frame.shape[1]),
                                             max(frame.shape[0], frame.shape[1])))
                tmp += [img]
            return tmp
        
        def create_xwyh_square(p_x1, p_x2, p_y1):
            x = int(self.faces[0].shape[1] * p_x1)
            w = int(self.faces[0].shape[1] * p_x2) - x
            y = int(self.faces[0].shape[0] * p_y1)
            h = w
            return x, y, w, h

        # y, x, z
        self.ROI_coords['full_face'] = [None, None, None, None]
        self.frames_ROI['full_face'] = create_ROI_(self.faces, *self.ROI_coords['full_face'])

        self.ROI_coords['forehead'] = [*create_xwyh_square(0.4, 0.6, 0.05)]
        self.frames_ROI['forehead'] = create_ROI_(self.faces, *self.ROI_coords['forehead'])

        self.ROI_coords['left_cheek'] = [*create_xwyh_square(0.15, 0.28, 0.5)]
        self.frames_ROI['left_cheek'] = create_ROI_(self.faces, *self.ROI_coords['left_cheek'])

        self.ROI_coords['right_cheek'] = [*create_xwyh_square(0.72, 0.85, 0.5)]
        self.frames_ROI['right_cheek'] = create_ROI_(self.faces, *self.ROI_coords['right_cheek'])


    def __apply_video_filters(self, idx_start, idx_finish):
        for key in self.frames_ROI:
            self.mean_g_signals[key] = get_avg_color_signal(self.frames_ROI[key][idx_start:idx_finish])
            self.frames_filtered[key] = add_filters_video(self.frames_ROI[key][idx_start:idx_finish], self.fps)
            self.mean_g_signals_filtered[key] = get_avg_color_signal(self.frames_filtered[key])


    def __process_signals(self):
        for key in self.mean_g_signals_filtered:
            detrend = signal.detrend(self.mean_g_signals_filtered[key])
            interpolated = np.interp(self.time_s, self.time_s, detrend)  # interpolation by 1
            interpolated = np.hamming(self.L) * interpolated  # make the signal become more periodic
            self.mean_g_signals_norm[key] = interpolated / np.linalg.norm(interpolated)


    def __get_fourier(self):
        self.fft = {}
        d_bpm = {}
        for key in self.mean_g_signals_norm:
            raw = np.fft.rfft(
                self.mean_g_signals_norm[key] * 30)  # do real fft with the normalization multiplied by 10
            freqs_ = float(self.fps) / self.L * np.arange(self.L // 2 + 1)
            freqs = 60. * freqs_

            fft = np.abs(raw) ** 2  # get amplitude spectrum
            # plot(freqs, fft, 'freq', 'Fourier before pruning')

            idx = np.where((freqs > 50) & (freqs < 180))  # the range of frequency that HR is supposed to be within
            pruned = fft[idx]
            pfreq = freqs[idx]

            freqs_ = pfreq
            fft = pruned
            # plot(pfreq, fft, 'freq', 'Fourierr after pruning')
            self.fft[key] = [pfreq, fft]
            idx2 = np.argmax(pruned)  # max in the range can be HR
            d_bpm[key] = int(freqs_[idx2])
        self.pulse_buff += [d_bpm]


    def __apply_filters_buff(self, idx_start, idx_finish):
        self.__apply_video_filters(idx_start, idx_finish)
        self.L = int(idx_finish - idx_start)
        self.time_s = np.array([1 / self.fps * i for i in range(self.L)])
        self.__process_signals()
        self.__get_fourier()
        self.times_s += [self.time_full[idx_start:idx_finish]]


    def process(self):
        def save_video(filename):
            out = cv2.VideoWriter(filename + '_pulse.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'),
                                  int(self.fps * self.save_speedx),
                                  (self.all_imgs[0].shape[1], self.all_imgs[0].shape[0]))
            print("========== Save File ==========")
            for im in tqdm(self.all_imgs):
                out.write(im)
            out.release()

        print("========== Createe ROI ==========")
        self.__create_ROI()

        #plt.imshow(self.faces[0])
        #plt.show()
        #plt.imshow(self.frames_ROI['full_face'][0])
        #plt.show()

        del self.frames
        self.fft_buff = []
        self.pulse_buff = []
        self.times_s = []
        self.time_full = np.array([1 / self.fps * i for i in range(len(self.faces))])
        if self.visualize: fig = plt.figure(figsize=(15, 10), dpi=100)
        print("========== Filters for buff Video ==========")
        self.all_imgs = []
        self.samples_ICA = []
        if len(self.faces) < self.buff_size: self.buff_size = len(self.faces)

        for max_idx in tqdm(range(30, len(self.faces))):
            self.__apply_filters_buff(max(max_idx - self.buff_size, 0), max_idx)
            if len(self.mean_g_signals_norm['forehead']) == self.buff_size:
                self.samples_ICA += [self.mean_g_signals['forehead']]
            if self.visualize: self.all_imgs += [self.__visualize__(fig, max_idx, self.frame_ids[max_idx])]
        if self.visualize: save_video(self.filename)
        else: return self.fft_buff


    def __visualize__(self, fig, max_idx, frame_id):
        def crop(img):
            # x, y
            return img[100:-50, 150:-100]

        def plot_graph_1(subplot, key):
            fig.add_subplot(*subplot)
            plt.plot(self.time_full[max(max_idx - self.buff_size, 0):max_idx],
                     self.mean_g_signals_filtered[key])
            plt.xticks(y_time,
                       rotation=30, fontsize=8)
            plt.yticks([])
            plt.legend([key], loc=1)

        # not use
        def plot_graph_2(subplot, key):
            fig.add_subplot(*subplot)
            plt.plot(self.time_full[max(max_idx - self.buff_size, 0):max_idx],
                     self.mean_g_signals_norm[key])
            plt.xticks(y_time,
                       rotation=30, fontsize=8)
            plt.yticks([])
            plt.legend([key], loc=1)

        def plot_graph_3(subplot, key):
            fig.add_subplot(*subplot)
            plt.plot(*self.fft[key])
            plt.legend([f"{self.pulse_buff[-1][key]} bpm"], loc=1)
            plt.yticks([])

        y_time = np.round(np.linspace(self.time_full[max(max_idx - self.buff_size, 0)],
                                      self.time_full[max_idx],
                                      10, ), 2)
        fig.add_subplot(3, 1, 1)
        plt.title(f"{frame_id}")
        plt.imshow(cv2.cvtColor(self.add_rectangles_img(self.faces[max_idx]), cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])

        for ii, key in enumerate(self.mean_g_signals_filtered):
            plot_graph_1([3, 4, 5 + ii], key)
        # for ii, key in enumerate(self.mean_g_signals_filtered):
        #     plot_graph_2([5, 4, 9 + ii], key)
        for ii, key in enumerate(self.mean_g_signals_filtered):
            plot_graph_3([3, 4, 9 + ii], key)

        canvas = FigureCanvasAgg(fig)
        s, (width, height) = canvas.print_to_buffer()
        X = np.fromstring(s, np.uint8).reshape((height, width, 4))
        # display(fig)
        plt.clf()
        # clear_output(wait=True)
        X = cv2.cvtColor(X, cv2.COLOR_RGBA2BGR)
        return crop(X)


if __name__ == "__main__":
    HRD = PulseAnalyzer(buff_size=150, save_speedx=0.2, filename='files/id2_0008.mp4', visualize=True)
    HRD.find_face_haard()
    HRD.process()
