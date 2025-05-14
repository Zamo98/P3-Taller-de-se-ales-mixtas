import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import soundfile as sf
import sounddevice as sd
import numpy as np
import threading

# Importar funciones para cada tipo de filtro
from fpb import generar_coeficientes_pasabajas, aplicar_filtro_iir as aplicar_fpb
from fpa import generar_coeficientes_pasaaltas
from fpbandas import generar_coeficientes_pasabandas
from fsp import generar_coeficientes_suprime_bandas

class FiltroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtro de Audio en Tiempo Real")

        self.file_label = tk.Label(root, text="Archivo: Ninguno")
        self.file_label.pack()

        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack()

        self.load_button = tk.Button(root, text="Cargar archivo", command=self.load_file)
        self.load_button.pack()

        self.play_button = tk.Button(root, text="Reproducir con filtro", command=self.play_filtered, state=tk.DISABLED)
        self.play_button.pack()

        self.stop_button = tk.Button(root, text="Detener reproducción", command=self.detener_reproduccion)
        self.stop_button.pack()

        self.filter_type_var = tk.StringVar(value="Pasa baja")
        self.filter_selector = ttk.Combobox(root, textvariable=self.filter_type_var,
                                            values=["Pasa baja configurable", "Pasa baja", "Pasa alta", "Pasa banda", "Suprime banda"]
                                           , state="readonly")
        self.filter_selector.pack()
        self.filter_selector.bind("<<ComboboxSelected>>", lambda e: self.on_filter_change())

        self.f1_container = tk.Frame(root)
        self.f1 = tk.Scale(self.f1_container, from_=100, to=10000, resolution=100, orient=tk.HORIZONTAL,
                           label="Frecuencia de corte (Hz)")
        self.f1.set(4000)
        self.f1.pack()
        self.f1.bind("<ButtonRelease-1>", lambda e: self.update_filter())

        self.data = None
        self.samplerate = None
        self.index = 0
        self.reproduciendo = False
        self.lock = threading.Lock()

        self.b = None
        self.a = None
        self.x_hist = None
        self.y_hist = None

    def load_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.file_label.config(text=f"Archivo: {filename}")
            self.status_label.config(text="Audio cargado. Listo para reproducir.")
            self.play_button.config(state=tk.NORMAL)

            self.data, self.samplerate = sf.read(filename)
            if self.data.ndim > 1:
                self.data = self.data[:, 0]
            self.index = 0

            self.update_filter()

    def get_filter_coefficients(self):
        if self.samplerate is None:
            return None, None

        filtro = self.filter_type_var.get()

        if filtro == "Pasa baja configurable":
            fc = self.f1.get()
            return generar_coeficientes_pasabajas(fc, self.samplerate)
        elif filtro == "Pasa baja":
            return generar_coeficientes_pasabajas(4000, self.samplerate)
        elif filtro == "Pasa alta":
            return generar_coeficientes_pasaaltas(8000, self.samplerate)
        elif filtro == "Pasa banda":
            return generar_coeficientes_pasabandas(5000, 12000, self.samplerate)
        elif filtro == "Suprime banda":
            return generar_coeficientes_suprime_bandas(4000, 8000, self.samplerate)
        else:
            return None, None

    def update_filter(self):
        with self.lock:
            b, a = self.get_filter_coefficients()
            if b is None or a is None:
                return
            self.b, self.a = b, a
            self.x_hist = np.zeros(len(b))
            self.y_hist = np.zeros(len(b))
            self.index = 0
            filtro = self.filter_type_var.get()
            self.status_label.config(text=f"{filtro} actualizado")

    def on_filter_change(self):
        filtro = self.filter_type_var.get()
        if filtro == "Pasa baja configurable":
            self.f1_container.pack()
        else:
            self.f1_container.pack_forget()
        self.update_filter()

    def callback(self, outdata, frames, time, status):
        if status:
            print(status)
        with self.lock:
            if not self.reproduciendo or self.data is None or self.b is None or self.a is None:
                outdata[:] = np.zeros((frames, 1))
                return

            total = len(self.data)
            end = min(self.index + frames, total)
            data_chunk = self.data[self.index:end]

            if len(data_chunk) < frames:
                padded = np.zeros(frames)
                padded[:len(data_chunk)] = data_chunk
                data_chunk = padded

            filtered, self.x_hist, self.y_hist = aplicar_fpb(data_chunk, self.b, self.a, self.x_hist, self.y_hist)
            outdata[:] = filtered.reshape(-1, 1)
            self.index += frames

    def play_filtered(self):
        if self.data is None:
            messagebox.showerror("Error", "Cargue un archivo de audio primero.")
            return

        self.index = 0
        self.x_hist = np.zeros(len(self.b))
        self.y_hist = np.zeros(len(self.b))
        self.reproduciendo = True
        self.status_label.config(text="Reproduciendo...")

        def reproducir():
            with sd.OutputStream(channels=1, callback=self.callback, samplerate=self.samplerate):
                while self.reproduciendo and self.index < len(self.data):
                    sd.sleep(100)

        threading.Thread(target=reproducir, daemon=True).start()

    def detener_reproduccion(self):
        self.reproduciendo = False
        self.index = 0
        self.status_label.config(text="Reproducción detenida.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FiltroApp(root)
    root.mainloop()
