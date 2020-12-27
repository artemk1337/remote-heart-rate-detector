from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt

from video_processing.main_algorithm import fourier_for_norm_signal


def apply_FastICA(mean_signals_norm, fps, n_components: int, filename='tmp'):
    assert n_components > 0

    #     plt.subplot(2, 1, 1)
    #     plt.bar(*np.unique(hearts_rate, return_counts=True), 5)

    pca = FastICA(n_components)
    H = pca.fit_transform(mean_signals_norm)

    plt.figure(figsize=(10, 5), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(H)
    plt.legend(['Signal'], loc=1)

    plt.subplot(1, 2, 2)
    if n_components == 1:
        fft, heart_rate = fourier_for_norm_signal(H, fps)
        plt.title(heart_rate)
        plt.plot(fft[0], fft[1])
    else:
        for i in range(n_components):
            fft, heart_rate = fourier_for_norm_signal(H[:, i], fps)
            plt.title(heart_rate)
            plt.plot(fft[0], fft[1])
        plt.legend(['Fourier'], loc=1)
    plt.savefig(filename + '.png')
    # plt.show()

    del pca
