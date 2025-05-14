import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, lfilter, lfilter_zi
import threading
import time

class RealTimeEqualizerLive:
    def __init__(self, root):
        self.root = root
        self.root.title("Ecualizador en Tiempo Real Interactivo")

        self.filename = None
        self.data = None
        self.samplerate = None
        self.stream = None
        self.block_size = 1024
        self.index = 0
        self.lock = threading.Lock()

        # Filtro en vivo
        self.b, self.a = None, None
        self.zi = None

        # Variables Tkinter
        self.filter_type_var = tk.StringVar()
        self.f1 = tk.DoubleVar(value=1000)

        self.create_widgets()

    def create_widgets(self):
        tk.Button(self.root, text="Cargar archivo WAV", command=self.load_file).pack(pady=10)

        ttk.Label(self.root, text="Tipo de filtro:").pack()
        self.filter_selector = ttk.Combobox(self.root, textvariable=self.filter_type_var, state="readonly")
        self.filter_selector['values'] = [
            "Pasa baja (fijo)",
            "Pasa baja (configurable)",
            "Pasa alta",
            "Suprime banda",
            "Pasa banda"
        ]
        self.filter_selector.current(1)
        self.filter_selector.pack()
        self.filter_selector.bind("<<ComboboxSelected>>", lambda e: self.update_filter())

        tk.Label(self.root, text="Frecuencia f1 (Hz)").pack()
        self.f1_slider = tk.Scale(self.root, from_=100, to=10000, resolution=100, orient="horizontal", variable=self.f1, command=lambda e: self.update_filter())
        self.f1_slider.pack(fill="x", padx=10)

        self.play_button = tk.Button(self.root, text="Reproducir con filtro en vivo", command=self.start_stream)
        self.play_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Detener", command=self.stop_stream)
        self.stop_button.pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.filename = file_path
            self.data, self.samplerate = sf.read(file_path)
            if self.data.ndim > 1:
                self.data = self.data[:, 0]  # Mono
            self.index = 0
            messagebox.showinfo("Archivo cargado", f"{self.filename}")

    def get_filter_coefficients(self):
        nyq = 0.5 * self.samplerate
        filter_type = self.filter_type_var.get()

        # Frecuencias definidas por requisito
        if filter_type == "Pasa baja (fijo)":
            cutoff = 4000  # Hz
            b, a = butter(5, cutoff / nyq, btype='low')

        elif filter_type == "Pasa baja (configurable)":
            cutoff = self.f1.get()
            b, a = butter(5, cutoff / nyq, btype='low')

        elif filter_type == "Pasa alta":
            cutoff = 8000  # Hz
            b, a = butter(5, cutoff / nyq, btype='high')

        elif filter_type == "Pasa banda":
            low = 5000
            high = 12000
            b, a = butter(5, [low / nyq, high / nyq], btype='bandpass')

        elif filter_type == "Suprime banda":
            low = 4000
            high = 8000
            b, a = butter(5, [low / nyq, high / nyq], btype='bandstop')

        else:
            # Fallback: Pasa baja 4kHz
            b, a = butter(5, 4000 / nyq, btype='low')

        return b, a


    def update_filter(self):
        with self.lock:
            self.b, self.a = self.get_filter_coefficients()
            self.zi = lfilter_zi(self.b, self.a) * self.data[self.index] if self.index < len(self.data) else np.zeros(len(self.b) - 1)

    def start_stream(self):
        if self.data is None:
            messagebox.showerror("Error", "Carga un archivo WAV primero.")
            return

        self.update_filter()

        def callback(outdata, frames, time_info, status):
            with self.lock:
                if self.index + self.block_size > len(self.data):
                    raise sd.CallbackStop

                chunk = self.data[self.index:self.index + self.block_size]
                filtered, self.zi = lfilter(self.b, self.a, chunk, zi=self.zi)
                outdata[:, 0] = filtered
                self.index += self.block_size

        self.stream = sd.OutputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            callback=callback
        )
        self.stream.start()

        # Hilo para monitorear y actualizar filtro si hay cambios
        threading.Thread(target=self.monitor_filter_changes, daemon=True).start()

    def monitor_filter_changes(self):
        prev_filter = (self.f1.get(), self.filter_type_var.get())
        while self.stream and self.stream.active:
            current_filter = (self.f1.get(), self.filter_type_var.get())
            if current_filter != prev_filter:
                self.update_filter()
                prev_filter = current_filter
            time.sleep(0.1)

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
