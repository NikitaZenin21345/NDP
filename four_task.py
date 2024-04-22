import numpy as np
import matplotlib.pyplot as plt
import timeit

def get_planck_taper_array(x):
    y2 = np.zeros(len(x))
    i = 0
    while i != len(x):
        y2[i] = planck_taper(i, len(x), 0.2)
        i = i + 1
    y2[0] = 0
    y2[len(x) - 1] = 0
    return y2


def planck_taper(x, N, E):
    if x == 0 or x == N - 1:
        return 0
    elif 0 < x < E * (N - 1):
        z = E * (N - 1) * (1 / x + 1 / (x - E * (N - 1)))
        return 1 / (np.exp(z) + 1)
    elif E * (N - 1) <= x <= (1 - E) * (N - 1):
        return 1.0
    elif (1 - E) * (N - 1) < x < N - 1:
        z = E * (N - 1) * (1 / (N - 1 - x) + 1 / ((1 - E) * (N - 1) - x))
        return 1 / (np.exp(z) + 1)


def plot_fourier_transform(signal, sample_rate):
    fft_result = np.fft.fft(signal)
    freq_axis = np.fft.fftfreq(len(signal), d=1 / sample_rate)
    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis[:len(signal) // 2], np.abs(fft_result)[: len(signal) // 2])
    plt.title('Спектр сигнала')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()



def convolution_sum(first_signal, second_signal):
    result_length = len(first_signal) + len(second_signal) - 1
    result = np.zeros(result_length, dtype=float)
    for i_index in range(result_length):
        element = 0.0
        for j_index in range(len(first_signal)):
            if 0 <= i_index - j_index < len(second_signal):
                element += first_signal[j_index] * second_signal[i_index - j_index]
        result[i_index] = element
    return result


def convolution_fft(first_signal, second_signal):
    result_len = first_signal.size + second_signal.size - 1
    fft_len = 2 ** int(np.ceil(np.log2(result_len)))
    fft_signal1 = np.fft.fft(first_signal, fft_len)
    fft_signal2 = np.fft.fft(second_signal, fft_len)
    fft_result = fft_signal1 * fft_signal2
    return np.fft.ifft(fft_result).real[:result_len]


def show_graph(duration, sampling_rate, x1, x2, y1, y2):
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.subplot(221)
    plt.title(' график сигнала ')
    plt.plot(x1, y1)
    plt.plot(x2, y2)

    result = convolution_sum(y1, y2)
    plt.subplot(222)
    plt.title(' график сигнала свертки sum')
    plt.plot(np.arange(0, len(result) / sampling_rate, 1 / sampling_rate), result)

    result = convolution_fft(y1, y2)
    plt.subplot(223)
    plt.title(' график сигнала свертки fft')
    plt.plot(np.arange(0, len(result) / sampling_rate, 1 / sampling_rate), result)

    result = np.convolve(y1, y2, mode='full')
    plt.subplot(224)
    plt.title(' график сигнала свертки встр.')
    plt.plot(np.arange(0, len(result) / sampling_rate, 1 / sampling_rate), result)
    plt.show()


end = 1
sampling_rate = 1000
x = np.arange(0, end, 1 / sampling_rate)
y1 = np.sign(np.sin(2 * np.pi * 5 * x))
# show_graph(end, sampling_rate, x, x, y1, y1)
y2 = np.sin(2 * np.pi * 5 * x)
# show_graph(end, sampling_rate, x, x, y2, y2)
y1 = np.sign(np.sin(2 * np.pi * 1 * x)) + 1
y2 = 2 * np.exp(-(x - 0.5) ** 2 / 0.1)
# show_graph(end, sampling_rate, x, x, y1, y2)
y2 = 4 * np.concatenate((x[:len(x) // 2 - 200], np.zeros(len(x) // 2 + 200)))
# show_graph(end, sampling_rate, x, x, y1, y2)
peakf = 0.5
fwdn = 0.5
s = fwdn * (2 * np.pi - 1) / (4 * np.pi)
y1 = np.sign(np.sin(2 * np.pi * 1 * x)) + np.random.normal(0, 0.1, len(x))
y2 = np.exp(-5 * ((x - peakf) / s) ** 2) * np.sin(2 * np.pi * x)
# plot_fourier_transform(y2, sampling_rate)
# show_graph(end, sampling_rate, x, x, y1, y2)


y2 = get_planck_taper_array(x)
#plot_fourier_transform(y2, sampling_rate)
#show_graph(end, sampling_rate, x, x, y1, y2)

y1 = 10 * np.sin(2 * 10 * np.pi * 1 * x) + np.random.normal(0, 0.1, len(x))
result = convolution_fft(y1, y2)
plt.subplot(223)
plt.title(' график сигнала свертки fft')
plt.plot(x, result[:len(x)])
y2 = np.exp(-5 * ((x - peakf) / s) ** 2) /10
result = convolution_fft(y1, y2)
plt.plot(x, result[:len(x)])
plt.plot(x, y1)
plt.show()