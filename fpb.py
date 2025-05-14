import numpy as np
from scipy.signal import cheby2, freqz
import matplotlib.pyplot as plt

# === 1. Generar coeficientes Chebyshev II ===
def generar_coeficientes_pasabajas(fc, fs, orden=2, atenuacion=40):
    b, a = cheby2(N=orden, rs=atenuacion, Wn=fc / (fs / 2), btype='low')
    return b, a

# === 2. Aplicar filtro con ecuación de diferencias ===
'''
def aplicar_filtro_iir(data, b, a, zi=None):
    N = len(b)
    x_hist = np.zeros(N)
    y_hist = np.zeros(N)

    if zi is not None:
        y_hist[:] = zi

    salida = np.zeros_like(data)

    for n in range(len(data)):
        x_hist[1:] = x_hist[:-1]
        x_hist[0] = data[n]

        y = 0.0
        for i in range(N):
            y += b[i] * x_hist[i]
        for i in range(1, N):
            y -= a[i] * y_hist[i - 1]

        y_hist[1:] = y_hist[:-1]
        y_hist[0] = y
        salida[n] = y

    return salida, y_hist.copy()
'''

import numpy as np

def aplicar_filtro_iir(data, b, a, x_hist, y_hist):
    N = len(b)
    salida = np.zeros_like(data)

    for n in range(len(data)):
        # Desplazar historial
        x_hist[1:] = x_hist[:-1]
        x_hist[0] = data[n]

        y = 0.0
        for i in range(N):
            y += b[i] * x_hist[i]
        for i in range(1, N):
            y -= a[i] * y_hist[i - 1]

        y_hist[1:] = y_hist[:-1]
        y_hist[0] = y

        salida[n] = y

    return salida, x_hist, y_hist

# === 3. Graficar espectro ===
def graficar_fft(signal, fs, titulo, fc=None):
    N = len(signal)
    freq = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(signal))
    espectro /= np.max(espectro)

    plt.plot(freq, espectro)
    plt.title(titulo)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud Normalizada")
    plt.xlim(0, 10000)
    plt.ylim(0, 1.05)
    if fc:
        plt.axvline(fc, color='red', linestyle='--', label=f'fc = {fc} Hz')
        plt.legend()
    plt.grid(True)

# === 4. Graficar respuesta en frecuencia ===
def graficar_respuesta_filtro(b, a, fs, fc=None):
    w, h = freqz(b, a, worN=8000)
    freqs = w * fs / (2 * np.pi)
    plt.plot(freqs, 20 * np.log10(abs(h)))
    plt.title("Respuesta en Frecuencia del Filtro (Bode)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.xlim(0, 10000)
    plt.ylim(-100, 5)
    if fc:
        plt.axvline(fc, color='red', linestyle='--', label=f'fc = {fc} Hz')
        plt.legend()
    plt.grid(True)
