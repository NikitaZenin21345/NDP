import numpy as np
from scipy.signal import butter, lfilter
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def calculate_transfer_function(poles_array, w, boundary_freq):
    transfer_function_array = np.zeros(w.size // 2, dtype=np.complex128)
    i = 0
    for freq in w[:len(w) // 2]:
        transfer_function = 1
        for pole in poles_array:
            denominator = 1j * freq / boundary_freq - pole
            if (denominator == 0):
                denominator = 0.0001 + 0j
            transfer_function *= 1 / (denominator)
        transfer_function_array[i] = transfer_function
        i += 1
    return np.concatenate((transfer_function_array, np.flip(transfer_function_array)))


def calculate_poles_array(level):
    array = np.arange(0, 2 * level)
    phase_array = np.pi * (-1 + 2 * array) / (2 * level)
    poles_array = np.array(2 * level - 1)
    for phase in phase_array:
        if 0.5 < phase % np.pi < 1.5:
            poles_array = np.append(poles_array, np.exp(1j * phase))
    return poles_array


def battervord_filter(yf, w, boundary_freq, level):
    N = 600
    end = 1
    x = np.linspace(0.0, end, N)
    xf = np.fft.fftfreq(x.size, d=end / N)
    poles_array = calculate_poles_array(level)
    transfer_function_array = calculate_transfer_function(poles_array, w, boundary_freq)
    plt.plot(xf[0:N // 2], np.abs(transfer_function_array[0:N // 2]) / max(transfer_function_array))
    plt.grid()
    plt.show()
    return np.abs(transfer_function_array) * yf


def battervord_filter_low(yf, w, boundary_freq, level):
    if level == 2:
        return np.abs((boundary_freq ** 2 ) / (-w ** 2 + 1j * np.sqrt(2) * boundary_freq * w + 1.0)) * yf


def battervord_filter_high(yf, w, boundary_freq, level):
    if level == 2:
        return np.abs((w ** 2 ) / (-boundary_freq ** 2 + 1j * np.sqrt(2) * boundary_freq * w + 1.0)) * yf


def battervord_filter_strip_1(yf, w, boundary_freq_left, boundary_freq_right, level):
    if level == 2:
        w0 = np.sqrt(boundary_freq_left * boundary_freq_right)
        return battervord_filter_low(yf, (w ** 2 + w0 ** 2) / ((boundary_freq_right - boundary_freq_left) * w), w0,
                                     level)


def battervord_filter_strip_2(yf, w, boundary_freq_left, boundary_freq_right, level):
    if level == 2:
        return battervord_filter_low(yf, w, boundary_freq_left, level) * battervord_filter_high(yf, w,
                                                                                                boundary_freq_right,
                                                                                                level)


def battervord_filter_blocking_1(yf, w, boundary_freq_left, boundary_freq_right, level):
    if level == 2:
        w0 = np.sqrt(boundary_freq_left * boundary_freq_right)
        return battervord_filter_low(yf, ((boundary_freq_right - boundary_freq_left) / (w ** 2 + w0 ** 2) * w), w0,
                                     level)


def battervord_filter_blocking_2(yf, w, boundary_freq_left, boundary_freq_right, level):
    if level == 2:
        return battervord_filter_low(yf, w, boundary_freq_left, level) + battervord_filter_high(yf, w,
                                                                                                boundary_freq_right,
                                                                                                level)


def print_filtered_signal(filter, boundary_freq, boundary_freq_right=None):
    N = 600
    end = 1
    x = np.linspace(0.0, end, N)
    y = np.sin(2 * np.pi * 5 * x) + np.sin(2 * np.pi * 45 * x)
    # y = y + np.random.normal(0, 0.1, x.shape)
    xf = np.fft.fftfreq(x.size, d=end / N)
    yf = fft(y)
    if boundary_freq_right is not None:
        yf_filtered = filter(fft(y), xf, boundary_freq, boundary_freq_right, 2)
    else:
        yf_filtered = filter(fft(y), xf, boundary_freq, 2)

    plt.figure(figsize=(6, 5))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.subplot(221)
    plt.title('а) график сигнала ')
    plt.plot(x, y / max(y))
    repaired_signal = np.fft.ifft(yf_filtered)
    plt.plot(x, repaired_signal / np.abs(np.max(repaired_signal)))
    plt.plot(x, np.sin(2 * np.pi * 5 * x))
    plt.grid()
    plt.subplot(223)
    plt.title('в) график спектра')
    plt.plot(xf[0:N // 2], (np.abs(yf[0:N // 2])) / np.abs(np.max(yf[0:N // 2])))
    plt.title('г) график спектра фильтрованного')
    filtered_max = np.max(np.abs(yf_filtered[0:N // 2]))
    plt.plot(xf[0:N // 2], np.abs(yf_filtered[0:N // 2]) / filtered_max)
    plt.grid()
    plt.show()

def battervord_comparison():
    N = 600
    end = 1
    boundary_freq = 20
    x = np.linspace(0.0, end, N)
    y = np.sin(2 * np.pi * 5 * x) + np.sin(2 * np.pi * 45 * x)
    #y = y + np.random.normal(0, 0.1, x.shape)
    xf = np.fft.fftfreq(x.size, d=end / N)
    yf = np.fft.fft(y)
    yf_filtered = np.fft.ifft(battervord_filter(np.fft.fft(y), xf, boundary_freq, 5))

    b, a = butter(5, boundary_freq, btype='low', fs=N)
    plt.figure(figsize=(6, 5))
    filtered_signal = lfilter(b, a, y)
    plt.plot(x, y/ np.abs(np.max(y)))
    plt.plot(x, (filtered_signal)/ np.abs(np.max(filtered_signal)))
    plt.plot(x, (yf_filtered)/ np.abs(np.max(yf_filtered)))

    plt.show()


#rint_filtered_signal(lambda yf, w, boundary_freq, level: battervord_filter_high(yf, w, boundary_freq, level), 20)
print_filtered_signal(lambda yf, w, boundary_freq, level: battervord_filter_low(yf, w, boundary_freq, level), 45)
# print_filtered_signal(lambda yf, w, boundary_freq, boundary_freq_right, level: battervord_filter_strip_2(yf, w, boundary_freq,boundary_freq_right, level), 45, 60)
# print_filtered_signal(lambda yf, w, boundary_freq,boundary_freq_right, level: battervord_filter_blocking_2(yf, w, boundary_freq,boundary_freq_right, level), 45,60)
# print_filtered_signal(lambda yf, w, boundary_freq, level: battervord_filter(yf, w, boundary_freq, 5), 15)
battervord_comparison()
