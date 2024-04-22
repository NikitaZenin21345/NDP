import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import timeit


def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def FFT(x):
    x = np.array(x, dtype=float)
    N = x.shape[0]

    if N % 2 != 0:
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)
    else:
        X0 = FFT(x[::2])
        X1 = FFT(x[1::2])
        W = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.hstack([X0 + W[:N // 2] * X1, X0 + W[N // 2:] * X1])


def first_task(N, x, y, xf):
    yff = fft(y)
    execution_time = timeit.timeit(lambda: fft(y), number=1)
    print(f"Время выполнения функции fft: {execution_time} секунд")

    yf = DFT_slow(y)
    execution_time = timeit.timeit(lambda: DFT_slow(y), number=1)
    print(f"Время выполнения функции DFT_slow: {execution_time} секунд")

    # пункт а
    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.title('а) график спектра')
    plt.plot(xf[0:N // 2], 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.xlabel('Частота, Гц (DFT_slow)')
    plt.ylabel('Амплитуда')
    plt.subplot(212)
    plt.plot(xf[0:N // 2], 2.0 / N * np.abs(yff[0:N // 2]))
    plt.grid()
    plt.xlabel('Частота, Гц (fft)')
    plt.ylabel('Амплитуда')
    plt.show()

    # # пункт б
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.subplot(222)
    # plt.plot(x, np.fft.ifft(yf))
    # plt.ylabel('Амплитуда (стало)');
    # plt.xlabel('Время');
    # plt.subplot(221)
    # plt.title('б) форма сигнала не изменилась')
    # plt.plot(x, y)
    # plt.ylabel('Амплитуда (было)');
    # plt.xlabel('Время ');
    # plt.show()
    #
    # # пункт в
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # y = y + np.random.normal(0, 0.1, x.shape)
    # yf = np.fft.fft(y)
    # plt.subplot(222)
    # plt.plot(x, np.fft.ifft(yf))
    # plt.ylabel('Амплитуда ');
    # plt.xlabel('Время (восстановленный сигнал)');
    # plt.subplot(221)
    # plt.title('в) с добавлением шума ')
    # plt.plot(x, y)
    # plt.ylabel('Амплитуда');
    # plt.xlabel('Время');
    # plt.subplot(223)
    # plt.title('спектр ')
    # plt.plot(xf[0:N // 2], 2.0 / N * np.abs(yf[0:N // 2]))
    # plt.ylabel('Амплитуда');
    # plt.xlabel('Частота ');
    # plt.show()


def second_task():
    N = 600
    freq = 5
    fmax = 1000
    T = 1.0 / fmax
    x = np.linspace(0.0, 4, N)
    y = 2 * np.sign(np.sin(2 * np.pi * freq * x))
    xf = np.fft.fftfreq(x.size, d=4/N)
    first_task(N, x, y, xf)


def third_task(N, x, y, xf):
    np.allclose(FFT(x), np.fft.fft(x))

    yff = fft(y)
    execution_time = timeit.timeit(lambda: fft(y), number=1)
    print(f"Время выполнения функции fft: {execution_time} секунд")

    yf = FFT(y)
    execution_time = timeit.timeit(lambda: FFT(y), number=1)
    print(f"Время выполнения функции FFT: {execution_time} секунд")

    plt.title('а) график спектра')
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.xlabel('Частота, Гц (FFT)');
    plt.ylabel('Амплитуда');
    plt.show()


N = 600
fmax = 1000
T = 1.0 / fmax
freq = 50
x = np.linspace(0.0, N * T, N)
y = np.cos(freq * 2.0 * np.pi * x) + np.cos(150 * 2.0 * np.pi * x)
xf = np.linspace(0.0, fmax / 2, N // 2)

#first_task(N, x, y, xf)
second_task()
#third_task(N, x, y, xf)
