import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def display_coefficient_on_freq_diagram(freq_array, coefficient_array, freq, time_step, coefficient_index):
    amplitude_array = np.zeros(coefficient_array.size//2)
    epsilon = 0.000000001
    i = 0
    for coefficients in coefficient_array:
        if (abs(coefficients[coefficient_index]) > epsilon):
            amplitude_array[i] = coefficients[0]
        i += 1
    plt.subplot(211)
    plt.grid()
    plt.xlabel('частота востановленного сигнала')
    plt.ylabel('амплитуда')
    plt.plot(freq_array, amplitude_array)


class fourier_series_transformer:
    def __init__(self):
        self.epsilon = 0.000000001

    def forward_transform(self, F, freq: float, precision: int = 20):
        a0 = 2 * freq * quad(lambda t: F(t), 0, 1 / freq)[0]
        coefficient_array = np.array([[a0 / 2, 0]])
        for i in range(1, precision):
            an = 2 * freq * quad(lambda t: F(t) * np.cos(2 * np.pi * freq * i * t), 0, 1 / freq)[0]
            bn = 2 * freq * quad(lambda t: F(t) * np.sin(2 * np.pi * freq * i * t), 0, 1 / freq)[0]
            if (an < self.epsilon and bn < self.epsilon):
                an = bn = 0
            new_array = np.array([an, bn])
            coefficient_array = np.append(coefficient_array, [new_array], axis=0)
        return coefficient_array

    def get_reverse_transformed_function(self, coefficient_array, time, freq: float, precision: int = 20):
        f = 0
        for i in range(0, precision):
            f += coefficient_array[i][0] * np.cos(2 * np.pi * freq * i * time) + \
                 coefficient_array[i][1] * np.sin(2 * np.pi * freq * i * time)
        return f

    def reverse_transform(self, coefficient_array, time_array, freq: float, precision: int = 20):
        f = []
        for time in time_array:
            f.append(self.get_reverse_transformed_function(coefficient_array, time, freq, precision))
        return f


def first_task():
    freq = 10
    time_array = np.arange(-1.0, 1.0, 0.002)
    F = np.sign(np.sin(2 * np.pi * freq * time_array))
    transformer = fourier_series_transformer()
    series_coeff = transformer.forward_transform(lambda t: np.sign(np.sin(2 * np.pi * freq * t)), freq)
    f = transformer.reverse_transform(series_coeff, time_array, freq)
    plt.subplot(211)
    plt.title('График функции')
    plt.plot(time_array, f)
    plt.plot(time_array, F)
    plt.xlabel('время')
    plt.ylabel('сигнал')
    plt.subplot(212)
    plt.plot(time_array, F - f)
    plt.xlabel('время')
    plt.ylabel('погрешность')
    plt.show()


def second_task():
    freq = 100
    time_step = 0.001
    time_array = np.arange(0, 1, time_step)
    F = np.cos(2 * np.pi * freq * time_array)

    transformer = fourier_series_transformer()
    series_coeff = transformer.forward_transform(lambda t: np.cos(2 * np.pi * freq * t), freq)
    f = transformer.reverse_transform(series_coeff, time_array, freq)
    plt.figure(figsize=(6, 5))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    freq_array = np.arange(0, 20, 1) * freq
    display_coefficient_on_freq_diagram(freq_array, series_coeff, freq, time_step, 0)
    f_reverse = np.fft.fft(F)
    plt.subplot(212)
    plt.grid()
    plt.xlabel('частота сигнала')
    plt.ylabel('амплитуда')
    freq_array = np.fft.fftfreq(time_array.size, d=time_step)
    plt.plot(freq_array[0:freq_array.size // 2], np.abs(f_reverse[0:f_reverse.size // 2]))
    plt.show()


def third_task():
    freq = 100
    time_step = 0.001
    time_array = np.arange(0, 1, time_step)
    F = np.sign(np.sin(2 * np.pi * freq * time_array))

    transformer = fourier_series_transformer()
    series_coeff = transformer.forward_transform(lambda t: np.sign(np.sin(2 * np.pi * freq * t)), freq)
    f = transformer.reverse_transform(series_coeff, time_array, freq)
    plt.figure(figsize=(6, 5))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    freq_array = np.arange(0, 20, 1) * freq
    display_coefficient_on_freq_diagram(freq_array, series_coeff, freq, time_step, 0)
    f_reverse = np.fft.fft(F)
    plt.subplot(212)
    plt.grid()
    plt.xlabel('частота сигнала')
    plt.ylabel('амплитуда')
    freq_array = np.fft.fftfreq(time_array.size, d=time_step)
    plt.plot(freq_array[0:freq_array.size // 2], np.abs(f_reverse[0:f_reverse.size // 2]))
    plt.show()


first_task()
second_task()
third_task()