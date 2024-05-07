import numpy as np
import matplotlib.pyplot as plt


def moving_average_filter(signal, window_size):
    filtered_signal = np.zeros(len(signal))
    for t in range(filtered_signal.size):
        sum = 0
        for i in range(t - window_size, t + window_size):
            if len(signal) > i >= 0:
                sum += signal[i]
        filtered_signal[t] = sum / (2 * window_size + 1)
    return filtered_signal


def delete_peak(array, max_peak):
    for index in range(len(array)):
        if np.abs(array[index]) > max_peak and index - 1 >= 0:
            array[index] = array[index - 1]
    return array


def median_filter(signal, kernel_size, max_peak):
    if kernel_size > len(signal):
        raise 'windows size larger signal size'
    n = len(signal)
    radius = kernel_size // 2
    filtered_signal = np.copy(signal)
    for i in range(n):
        if i - radius > 0 and i + radius < n:
            window = signal[i - radius:i + radius]
        else:
            if i - radius <= 0:
                window = signal[:i + radius]
            elif i + radius >= n:
                window = signal[i - radius:n - 1]
        window = delete_peak(window, max_peak)
        filtered_signal[i] = np.sort(window)[len(window) // 2]
    return filtered_signal


def gaussian_smoothing_filter(x, time, window_size, fwhm=25.4):
    g = np.exp(-4 * np.log(2) * (time ** 2) / fwhm)
    y = np.zeros(len(x))
    for t in range(y.size - 1):
        sum = 0
        for i in range(t - window_size, t + window_size):
            if i < len(x):
                sum += x[i] * g[i]
        y[t] = sum / (2 * window_size + 1)
    return y


def show_signal_and_spectrum(sampling_rate, x1, y1, position1=221, position2=222, title=''):
    plt.subplot(position1)
    plt.title('график сигнала ' + title)
    plt.xlabel('время')
    plt.ylabel('Амплитуда')
    plt.plot(x1, y1)
    plt.subplot(position2)
    plt.title('график спектра ' + title)
    plt.xlabel('частота')
    plt.ylabel('Амплитуда')
    plt.plot(np.fft.fftfreq(x1.size, d=1 / sampling_rate)[0:x1.size // 2],
             (np.abs(np.fft.fft(y1))[0:y1.size // 2]) / (np.max(np.abs(np.fft.fft(y1)))))


# __________________________________________________________
end = 10
sampling_rate = 1000
x = np.arange(0, end, 1 / sampling_rate)
y = np.sin(2 * np.pi * 10 * x) + np.random.normal(0, 0.1, len(x))


plt.figure(figsize=(6, 6))
plt.subplots_adjust(wspace=0.4, hspace=0.4)


show_signal_and_spectrum(sampling_rate, x, y, title='average')
show_signal_and_spectrum(sampling_rate, x, moving_average_filter(y, 20))
plt.show()
show_signal_and_spectrum(sampling_rate, x, y, title='gaussian')
show_signal_and_spectrum(sampling_rate, x, gaussian_smoothing_filter(y, x, 4, 1000))
plt.show()
# __________________________________________________________

def generate_signal_with_random_peak_value(num_peaks, signal_length, peak_mean=0.5, peak_std=1):
    signal = np.zeros(signal_length)
    peak_indices = np.random.randint(0, signal_length, num_peaks)
    peak_values = np.random.uniform(0, 1, signal_length)
    for index, value in zip(peak_indices, peak_values):
        signal[index] += value
    return signal


def generate_peak_signal(num_peaks, signal_length, peak_height=1, peak_width=10):
    signal = np.zeros(signal_length)
    peak_indices = np.random.randint(0, signal_length, num_peaks)
    for index in peak_indices:
        signal[index] += peak_height
    return signal


# __________________________________________________________
#num_peaks = 50
#peak_height = 1
#peak_width = 5

#y = np.random.normal(scale=0.2, size=len(x))
#signal = (0.1 - generate_peak_signal(num_peaks, len(y), peak_height, peak_width)) * y

#show_signal_and_spectrum(sampling_rate, x, signal, title='gaussin on random peak with constant height')
#show_signal_and_spectrum(sampling_rate, x, gaussian_smoothing_filter(signal, x, 4, 1000))
#plt.show()

#y = np.abs(y)
#signal = (0.1 - generate_signal_with_random_peak_value(num_peaks, len(y), peak_height, peak_width)) * y


#show_signal_and_spectrum(sampling_rate, x, signal, title='median on random peak with random height')
#show_signal_and_spectrum(sampling_rate, x, median_filter(signal, 3, peak_height))
#plt.show()
# __________________________________________________________

# def calculate_bic(signal, predicted_signal, k):
#     n = len(signal)
#     bic = n * np.log(np.sum((signal - predicted_signal) ** 2) / n) + k * np.log(n)
#     return bic

#
# def plot_bics_and_coeffs(bics):
#     bics_parameter = list(bics.keys())
#     coeff = list(bics.values())
#     plt.figure(figsize=(8, 6))
#     plt.bar(coeff, bics_parameter, color='skyblue')
#     plt.xlabel('coeff')
#     plt.ylabel('bics_parameter')
#     plt.title('Approximation')
#     min_value = bics[min(bics.keys())]
#     if min_value is not None:
#         plt.axvline(min_value, color='red', linestyle='--', label=f'Min Value {min_value}')
#         plt.legend()
#     plt.show()
#
#
# def find_linear_trend_parameters(signal, time):
#     bics = {}
#     coeffs = np.arange(0, 10, 1 / 1000)
#     for coeff in coeffs:
#         linear_trend = coeff * time
#         bics[calculate_bic(signal, linear_trend, 1)] = coeff
#     best_bic = min(bics.keys())
#     plot_bics_and_coeffs(bics)
#     return bics[best_bic] * time
#

# __________________________________________________________
# time = np.linspace(0, 100, 100)
#
# signal = 0.5 * time + np.random.normal(size=time.size) + 1
# linear_trend = find_linear_trend_parameters(signal, time)
# plt.plot(time, signal, label='signal')
# plt.plot(time, linear_trend, label='linear_trend')
# plt.plot(time, signal - linear_trend, label='signal - linear_trend')
# plt.legend()
# plt.show()
# # __________________________________________________________
#
# y = np.random.normal(scale=0.2, size=len(time))
# signal = np.abs(generate_signal_with_random_peak_value(num_peaks//2 - 10, len(y), peak_height, peak_width)) + np.abs(y)
# windows_size = len(signal) * 0.05
# mean_time_series = np.ones(len(time), dtype=float) * np.mean(signal)
# std3_time_series = np.ones(len(time), dtype=float) * np.std(signal)
# treshold = mean_time_series + std3_time_series
# outliers = (signal > treshold)
# plt.plot(time, signal, 'k', label='Signal')
# plt.plot(time,treshold , 'm--', label='Treshold ')
# plt.plot(time[outliers], signal[outliers], 'ro', label='Outliers')
# index = 0
# for outline in outliers:
#     if index - 1 > 0 and index + 1 < len(signal):
#         if outline:
#             signal[index] = (signal[index - 1] + signal[index + 1]) / 2
#     index += 1
# plt.plot(time, signal, label='Filtered signal')
# plt.legend()
# plt.show()
# ____________________________________________________


time = np.linspace(0, 10, 1000)
noise_intervals = [(200, 300), (700, 800)]
noise_level = 5
signal = 2 * np.sin(2 * np.pi * time) + 0.5 * np.random.normal(size=time.size)

noisy_signal = np.copy(signal)
for start, end in noise_intervals:
    noisy_signal[start:end] += 2 * np.random.normal(0, noise_level, end - start)
signal = noisy_signal
windows_size = len(signal) * 0.005
rms_time_series = np.zeros(len(signal))
for index in range(0, len(rms_time_series)):
    low_bnd = int(np.max((0, index - windows_size)))
    upp_bnd = int(np.min((index + windows_size, len(signal))))
    tmp_signal = signal[low_bnd:upp_bnd] - np.mean(signal[low_bnd:upp_bnd])
    rms_time_series[index] = np.sqrt(np.sum(tmp_signal ** 2))

treshold = 15
outliers = (rms_time_series > treshold)

plt.subplot(211)
plt.plot(time, treshold * np.ones(time.size), 'm--', label='Treshold ')
plt.plot(time[outliers], rms_time_series[outliers], 'ro', label='Outliers')
plt.plot(time, rms_time_series, label='RMS ')
plt.legend()
plt.subplot(212)
plt.plot(time, signal, label='Signal ')
index = 0
for outline in outliers:
    if index - 1 > 0 and index + 1 < len(signal):
        if outline:
            signal[index] = None
    index += 1
plt.plot(time, signal, label='Filtered signal')
plt.legend()
plt.show()
# ______________________________________________________
def spectral_interpolation(signal, missing_index, window_size):

    start, end = missing_index
    n = len(signal)
    modified_signal = np.copy(signal)
    window_pre = signal[max(0, start - window_size):start]
    window_post = signal[end:min(n, end + window_size)]

    spectrum_pre = np.fft.fft(window_pre, n=window_size)
    spectrum_post = np.fft.fft(window_post, n=window_size)
    average_spectrum = (spectrum_pre + spectrum_post) / 2
    restored_part = np.fft.ifft(average_spectrum).real
    modified_signal[start:end] = restored_part[:end - start]

    if start != 0:
        modified_signal[start:start + 1] = (modified_signal[start - 1] + modified_signal[start]) / 2
    if end < n:
        modified_signal[end - 1:end] = (modified_signal[end - 1] + modified_signal[end]) / 2

    return modified_signal


np.random.seed(0)
time = np.linspace(0, 10 * np.pi, 1000)
original_signal = np.sin(time) + np.random.normal(0, 0.5, time.size)
missing_start, missing_end = 200, 250
corrupted_signal = np.copy(original_signal)
corrupted_signal[missing_start:missing_end] = None
restored_signal = spectral_interpolation(corrupted_signal, (missing_start, missing_end), 50)


plt.figure(figsize=(12, 6))
plt.plot(original_signal,  label='Оригинальный сигнал', alpha=0.7)
plt.plot(corrupted_signal, 'bo', label='Сигнал с пропущенными данными', alpha=0.7)
plt.plot(restored_signal, '--', label='Восстановленный сигнал', alpha=0.7)
plt.legend()
plt.title('Спектральная интерполяция сигналов')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.show()
#___________________________________________________________________________

#___________________________________________________________________
# from scipy.signal import butter, filtfilt
# from scipy.interpolate import griddata
#
# def decimate_signal(signal, fs, fs_new):
#     decimation_factor = fs // fs_new
#     f_cut = fs_new / 2
#     nyquist = f_cut / 2
#     b, a = butter(5, nyquist / (fs / 2), btype='low')
#     filtered_signal = filtfilt(b, a, signal)
#     decimated_signal = filtered_signal[::decimation_factor]
#     return decimated_signal
#
#
# def upsample_signal(t, signal, fs_new):
#     duration = t[-1] - t[0] + (t[1] - t[0])
#     t_new = np.linspace(0, duration, int(duration * fs_new), endpoint=False)
#     signal_new = griddata(t, signal, t_new, method='cubic')
#     return t_new, signal_new
#
#
# def resample(t, signal, fs, fs_new):
#     if fs >= fs_new:
#         return decimate_signal(signal, fs, fs_new)
#     else:
#         return upsample_signal(t, signal, fs_new)
#
#
# plt.figure(figsize=(12, 6))
# fs = 500
# t = np.linspace(0, 1, fs, endpoint=False)
# signal = np.sin(2 * np.pi * 100 * t)
# plt.subplot(2, 1, 1)
# plt.plot(t, signal, '-og', label='Original Signal')
#
# fs = 1000
# fs_new = 500
# t1 = np.linspace(0, 1, fs, endpoint=False)
# signal1 = np.sin(2 * np.pi * 50 * t1) + np.sin(2 * np.pi * 200 * t1)
# plt.plot(t1, signal1,'-ob', label='Downsampled Signal')
#
# downsampled_signal = resample(t1, signal1, fs, fs_new)
# t_downsampled = np.linspace(0, 1, len(downsampled_signal), endpoint=False)
#
# fs = 250
# t2 = np.linspace(0, 1, fs, endpoint=False)
# signal2 = np.sin(2 * np.pi * 30 * t2)
# t_upsampled, upsampled_signal = resample(t2, signal2, fs, fs_new)
# plt.plot(t2, signal2,'-or', label='Upsampled Signal')
# plt.legend()
# plt.xlim(0.0, 0.1)
#
# plt.title('Original Signal')
# plt.subplot(2, 1, 2)
# plt.plot(t, signal, '-og', label='Original Signal')
# plt.plot(t_upsampled, downsampled_signal, '-ob', label='Downsampled Signal')
# plt.plot(t_upsampled, upsampled_signal, '-or', label='Upsampled Signal')
# plt.title('Upsampled Signal')
# plt.xlim(0.0, 0.1)
# plt.legend()
# plt.show()








