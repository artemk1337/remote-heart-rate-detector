import scipy.fftpack as fftpack
import numpy as np
import cv2


def build_gaussian_pyramid(src,level=3):
    s=src.copy()
    pyramid=[s]
    for i in range(level):
        s=cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid


def gaussian_video(video_tensor,levels=3):
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
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
    final_video=np.zeros(origin_video.shape)
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img=cv2.resize(img, (origin_video[i].shape[1], origin_video[i].shape[0]))
        img=img+origin_video[i]
        final_video[i]=img
    return final_video


def magnify_color(data_buffer,fps,low=0.4,high=2,levels=3,amplification=30):
    gau_video=gaussian_video(data_buffer,levels=levels)
    filtered_tensor=temporal_ideal_filter(gau_video,low,high,fps)
    amplified_video=amplify_video(filtered_tensor,amplification=amplification)
    final_video=reconstract_video(amplified_video,data_buffer,levels=levels)
    #print("c")
    return final_video
