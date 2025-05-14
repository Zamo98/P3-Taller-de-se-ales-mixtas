import numpy as np
from scipy.signal import cheby2

# Generar coeficientes Pasa Altas
def generar_coeficientes_pasaaltas(fc, fs, orden=4, atenuacion=40):
    b, a = cheby2(N=orden, rs=atenuacion, Wn=fc/(fs/2), btype='high')
    return b, a

# Aplicar filtro
def aplicar_filtro_iir(data, b, a):
    N = len(b)
    x_hist = np.zeros(N)
    y_hist = np.zeros(N)
    salida = np.zeros_like(data)

    for n in range(len(data)):
        x_hist[1:] = x_hist[:-1]
        x_hist[0] = data[n]

        y = 0.0
        for i in range(N):
            y += b[i] * x_hist[i]
        for i in range(1, N):
            y -= a[i] * y_hist[i-1]

        y_hist[1:] = y_hist[:-1]
        y_hist[0] = y

        salida[n] = y

    return salida
