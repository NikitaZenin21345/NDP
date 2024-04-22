import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.io import wavfile
from scipy.signal import spectrogram
def haar_function(t):
    if 0 <= t <= 1/2:
        return 1
    if 1/2 < t <= 1:
        return -1
    return 0


def haar(x):
    result = np.zeros(x.size)
    for i in range(x.size):
        result[i] = haar_function(x[i])
    return result


def show_signal_and_spectrum(sampling_rate, x1, y1, position1 = 211, position2 = 212 ):
    plt.subplot(position1)
    plt.title('график сигнала')
    plt.xlabel('время')
    plt.ylabel('Амплитуда')
    plt.plot(x1, y1)
    plt.subplot(position2)
    plt.title('график сигнала')
    plt.xlabel('частота')
    plt.ylabel('Амплитуда')
    plt.plot(np.fft.fftfreq(x1.size, d = 1 / sampling_rate)[0:x1.size//2], (np.abs(np.fft.fft(y1))[0:y1.size//2])/(np.max(np.abs(np.fft.fft(y1)))))


def show_convolution_singnals(sampling_rate, x, y1, y2):
    result = np.convolve(y1, y2, mode='full')
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    show_signal_and_spectrum(sampling_rate, x, y1, 221, 222)
    show_signal_and_spectrum(sampling_rate, np.arange(0, len(result) / sampling_rate, 1 / sampling_rate), result/np.max(result), 223, 222)
    plt.show()


def show_wavelet_filtretion(sampling_rate, x, y):
    morle = np.exp(-0.5 * x ** 2) * np.cos(5 * x)
    show_convolution_singnals(sampling_rate, x, y, morle)
    hat = np.exp(-0.5 * x ** 2) * (1 - x ** 2)
    show_convolution_singnals(sampling_rate, x, y, hat)
    show_convolution_singnals(sampling_rate, x, y, haar(x))

def plot_spectrogram(y, fs, ylim):
    plt.figure(figsize=(10, 6))
    plt.specgram(y, Fs=fs, NFFT=1024, noverlap=512, cmap='jet')
    plt.title('Spectrogram of Signal')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([0, ylim])
    plt.xlabel('Time (s)')
    plt.colorbar(label='Intensity (dB)')
    plt.show()

end = 10
sampling_rate = 1000
# x = np.arange(0, end, 1 / sampling_rate)
# #Морле

# y1 = np.exp(-0.5 * x**2) * np.cos(5 * x)
# show_signal_and_spectrum(sampling_rate, x, y1)
# plt.show()
# #Шляпа
# y1 = np.exp(-0.5 * x**2 /0.01) * (1 - x**2)
# show_signal_and_spectrum(sampling_rate, x, y1)
# plt.show()
# #Haar
# y1 = haar(x)
# show_signal_and_spectrum(sampling_rate, x, y1)
# plt.show()
# #____________________________________________________________________________
# y1 = np.cos(10 * 2.0 * np.pi * x) + np.cos(5 * 2.0 * np.pi * x) + np.random.normal(0, 0.1, x.shape)
# show_wavelet_filtretion(sampling_rate, x, y1)

#____________________________________________________________________________
t = np.arange(-4, 4, 1/sampling_rate)
N = len(t)

freqmod = np.exp(-t ** 2)*10+10
freqmod = freqmod + np.linspace(0, 10, N)
signal = np.sin(2 * np.pi * (t + np.cumsum(freqmod)/sampling_rate))
#plt.plot(t, signal)
#plt.show()

morle_wavelet = np.exp(2j* np.pi * t) * np.exp(-(t**2)/0.1)
filtered_signal = np.convolve(signal, morle_wavelet, mode='same')

plot_spectrogram(signal, sampling_rate, 151)

#__________________________________________________
sampling_rate, samples = wavfile.read('zvuk-notyi-lya.wav')
samples = samples[:,0] / np.max(samples[:,0])
N = len(samples)
t = np.linspace(0, N, sampling_rate)


morlet = np.exp(-0.5 * t**2) * np.cos(5 * t)
hat = np.exp(-0.5 * t**2) * (1 - t**2)
haar_ = haar(t)

morlet_filtered_signal = np.convolve(samples, morlet)[:-1000]
hat_filtered_signal = np.convolve(samples, hat)[:-1000]
haar_filtered_signal = np.convolve(samples, haar_)[:-1000]
morlet_filtered_signal_spectrum = np.abs(np.fft.fft(morlet_filtered_signal))
hat_filtered_signal_spectrum = np.abs(np.fft.fft(hat_filtered_signal))
haar_filtered_signal_spectrum = np.abs(np.fft.fft(haar_filtered_signal))
freq = np.fft.fftfreq(len(morlet_filtered_signal), 1/sampling_rate)
plt.plot(freq, morlet_filtered_signal_spectrum, label='morle')
plt.plot(freq, hat_filtered_signal_spectrum, label='hat')
plt.plot(freq, haar_filtered_signal_spectrum, label='haar')
plt.legend()
plt.ylabel('амплитуда')
plt.xlabel('частота(Гц)')
plt.show()

plt.figure(figsize=(12, 8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.subplot(221)
plt.specgram(samples, Fs=sampling_rate, NFFT=1024, noverlap=512, cmap='jet')
plt.title('Spectrogram of original')
plt.ylabel('Frequency (Hz)')
plt.ylim([0, sampling_rate//2])
plt.xlim([0,  0.493])
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity (dB)')
plt.subplot(222)
plt.specgram(morlet_filtered_signal, Fs=sampling_rate, NFFT=1024, noverlap=512, cmap='jet')
plt.title('Spectrogram of morle')
plt.ylabel('Frequency (Hz)')
plt.ylim([0, sampling_rate//2])
plt.xlim([0,  0.493])
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity (dB)')
plt.subplot(223)
plt.specgram(hat_filtered_signal, Fs=sampling_rate, NFFT=1024, noverlap=512, cmap='jet')
plt.title('Spectrogram of hat')
plt.ylabel('Frequency (Hz)')
plt.ylim([0, sampling_rate//2])
plt.xlim([0, 0.493])
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity (dB)')
plt.subplot(224)
plt.specgram(haar_filtered_signal, Fs=sampling_rate, NFFT=1024, noverlap=512, cmap='jet')
plt.title('Spectrogram of haar')
plt.ylabel('Frequency (Hz)')
plt.ylim([0, sampling_rate//2])
plt.xlim([0,  0.493])
plt.xlabel('Time (s)')
plt.colorbar(label='Intensity (dB)')
plt.show()

