import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
import time

from fpb import generar_coeficientes_pasabajas, aplicar_filtro_iir

class RealTimeEqualizerLive:
    def __init__(self, root):
        self.root = root
        self.root.title("Ecualizador Digital")

        self.filename = None
        self.data = None
        self.samplerate = None
        self.stream = None
        self.block_size = 1024
        self.index = 0
        self.lock = threading.Lock()

        self.b, self.a = None, None
        self.zi = None

        self.filter_type_var = tk.StringVar()
        self.f1 = tk.DoubleVar(value=4000)  # Frecuencia de corte configurable

        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.root, text="Cargar archivo WAV", command=self.load_file).pack(pady=10)

        ttk.Label(self.root, text="Tipo de filtro:").pack()
        self.filter_selector = ttk.Combobox(self.root, textvariable=self.filter_type_var, state="readonly")
        self.filter_selector['values'] = [
            "Pasa baja (fijo)",
            "Pasa baja (configurable)",
        ]
        self.filter_selector.current(0)
        self.filter_selector.pack()
        self.filter_selector.bind("<<ComboboxSelected>>", lambda e: self.update_filter())

        tk.Label(self.root, text="Frecuencia de corte (Hz)").pack()
        self.f1_slider = tk.Scale(self.root, from_=100, to=10000, resolution=100, orient="horizontal", variable=self.f1, command=lambda e: self.update_filter())
        self.f1_slider.pack(fill="x", padx=10)

        self.play_button = tk.Button(self.root, text="Reproducir con filtro", command=self.start_stream)
        self.play_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Detener", command=self.stop_stream)
        self.stop_button.pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.filename = file_path
            self.data, self.samplerate = sf.read(file_path)
            if self.data.ndim > 1:
                self.data = self.data[:, 0]
            self.index = 0
            messagebox.showinfo("Archivo cargado", f"{self.filename}")

    def get_filter_coefficients(self):
        filter_type = self.filter_type_var.get()
        fc = 4000 if filter_type == "Pasa baja (fijo)" else self.f1.get()
        b, a = generar_coeficientes_pasabajas(fc, self.samplerate)
        return b, a

    def update_filter(self):
        with self.lock:
            self.b, self.a = self.get_filter_coefficients()
            if self.index < len(self.data):
                self.zi = np.zeros(len(self.b) - 1)

    def start_stream(self):
        if self.data is None:
            messagebox.showerror("Error", "Carga un archivo WAV primero.")
            return

        self.update_filter()
        zi = np.zeros(len(self.b))

        def callback(outdata, frames, time_info, status):
            nonlocal zi
            with self.lock:
                if self.index + self.block_size > len(self.data):
                    raise sd.CallbackStop

                chunk = self.data[self.index:self.index + self.block_size]
                filtrado, zi = aplicar_filtro_iir(chunk, self.b, self.a, zi=zi)
                outdata[:, 0] = filtrado
                self.index += self.block_size

        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            callback=callback
        )
        self.stream.start()

        threading.Thread(target=self.monitor_filter_changes, daemon=True).start()

    def monitor_filter_changes(self):
        prev_filter = (self.f1.get(), self.filter_type_var.get())
        while self.stream and self.stream.active:
            current_filter = (self.f1.get(), self.filter_type_var.get())
            if current_filter != prev_filter:
                self.update_filter()
                prev_filter = current_filter
            time.sleep(0.2)

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.index = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeEqualizerLive(root)
    root.mainloop()
