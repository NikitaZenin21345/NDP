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


# show_signal_and_spectrum(sampling_rate, x, y, title='average')
# show_signal_and_spectrum(sampling_rate, x, moving_average_filter(y, 4))
# plt.show()
# show_signal_and_spectrum(sampling_rate, x, y, title='gaussian')
# show_signal_and_spectrum(sampling_rate, x, gaussian_smoothing_filter(y, x, 4, 1000))
# plt.show()
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
num_peaks = 50
peak_height = 1
peak_width = 5

y = np.random.normal(scale=0.2, size=len(x))
signal = (0.1 - generate_peak_signal(num_peaks, len(y), peak_height, peak_width)) * y

# show_signal_and_spectrum(sampling_rate, x, signal, title='gaussin on random peak with constant height')
# show_signal_and_spectrum(sampling_rate, x, gaussian_smoothing_filter(signal, x, 4, 1000))
# plt.show()

y = np.abs(y)
signal = (0.1 - generate_signal_with_random_peak_value(num_peaks, len(y), peak_height, peak_width)) * y


# show_signal_and_spectrum(sampling_rate, x, signal, title='median on random peak with random height')
# show_signal_and_spectrum(sampling_rate, x, median_filter(signal, 3, peak_height))
# plt.show()
# __________________________________________________________

def calculate_bic(signal, predicted_signal, k):
    n = len(signal)
    bic = n * np.log(np.sum((signal - predicted_signal) ** 2) / n) + k * np.log(n)
    return bic


def plot_bics_and_coeffs(bics):
    bics_parameter = list(bics.keys())
    coeff = list(bics.values())
    plt.figure(figsize=(8, 6))
    plt.bar(coeff, bics_parameter, color='skyblue')
    plt.xlabel('coeff')
    plt.ylabel('bics_parameter')
    plt.title('Approximation')
    min_value = bics[min(bics.keys())]
    if min_value is not None:
        plt.axvline(min_value, color='red', linestyle='--', label=f'Min Value {min_value}')
        plt.legend()
    plt.show()


def find_linear_trend_parameters(signal, time):
    bics = {}
    coeffs = np.arange(0, 10, 1 / 1000)
    for coeff in coeffs:
        linear_trend = coeff * time
        bics[calculate_bic(signal, linear_trend, 1)] = coeff
    best_bic = min(bics.keys())
    plot_bics_and_coeffs(bics)
    return bics[best_bic] * time


# __________________________________________________________
time = np.linspace(0, 100, 100)


# signal = 0.5 * time + np.random.normal(size=time.size) + 1
# linear_trend = find_linear_trend_parameters(signal, time)
# plt.plot(time, signal, label='signal')
# plt.plot(time, linear_trend, label='linear_trend')
# plt.plot(time, signal - linear_trend, label='signal - linear_trend')
# plt.legend()
# plt.show()
# __________________________________________________________

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

def clean_signal(signal, threshold):
    clean_signal = np.copy(signal)
    clean_signal[np.abs(signal) > threshold] = 0
    return clean_signal


def sliding_rms(signal, window_size):
    rms = np.sqrt(np.convolve(signal ** 2, np.ones(window_size) / window_size, mode='same'))
    return rms


time = np.linspace(0, 10, 1000)
noise_intervals = [(200, 300), (700, 800)]
noise_level = 5
signal = 2 * np.sin(2 * np.pi * time) + 0.5 * np.random.normal(size=time.size)

noisy_signal = np.copy(signal)
for start, end in noise_intervals:
    noisy_signal[start:end] += np.random.normal(0, noise_level, end - start)
plt.plot(noisy_signal)
plt.show()
