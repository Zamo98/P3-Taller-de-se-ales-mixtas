import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import soundfile as sf
import sounddevice as sd
import numpy as np
import threading

from fpb import generar_coeficientes_pasabajas, aplicar_filtro_iir as aplicar_fpb
from fpa import generar_coeficientes_pasaaltas
from fpbandas import generar_coeficientes_pasabandas
from fsp import generar_coeficientes_suprime_bandas

class FiltroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtro de Audio en Tiempo Real")
        self.root.configure(bg="#000000")

        # === Estilos ttk ===
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", foreground="white", background="#3CD31A", font=("Segoe UI", 10), padding=6)
        style.configure("TLabel", foreground="white", background="#000000", font=("Segoe UI", 10))
        style.configure("TCombobox", padding=4)

        # === Información del archivo ===
        self.file_label = ttk.Label(root, text="Archivo: Ninguno")
        self.file_label.pack(pady=(10, 5))

        self.status_label = ttk.Label(root, text="")
        self.status_label.pack()

        # === Controles ===
        controls_frame = ttk.Frame(root)
        controls_frame.pack(pady=10)

        self.load_button = ttk.Button(controls_frame, text="Cargar archivo", command=self.load_file)
        self.load_button.grid(row=0, column=0, padx=5)

        self.play_button = ttk.Button(controls_frame, text="Reproducir con filtro", command=self.play_filtered, state=tk.DISABLED)
        self.play_button.grid(row=0, column=1, padx=5)

        self.stop_button = ttk.Button(controls_frame, text="Detener reproducción", command=self.detener_reproduccion)
        self.stop_button.grid(row=0, column=2, padx=5)

        # === Selector de filtro ===
        selector_frame = ttk.Frame(root)
        selector_frame.pack(pady=(5, 10))

        ttk.Label(selector_frame, text="Tipo de filtro:").grid(row=0, column=0, sticky='w', padx=5)
        self.filter_type_var = tk.StringVar(value="Pasa baja")
        self.filter_selector = ttk.Combobox(selector_frame, textvariable=self.filter_type_var,
                                            values=["Pasa baja configurable", "Pasa baja", "Pasa alta", "Pasa banda", "Suprime banda"],
                                            state="readonly", width=25)
        self.filter_selector.grid(row=0, column=1, padx=5)
        self.filter_selector.bind("<<ComboboxSelected>>", lambda e: self.on_filter_change())

        # === Control de frecuencia de corte ===
        self.f1_container = ttk.Frame(root)
        self.f1 = tk.Scale(self.f1_container, from_=100, to=10000, resolution=100, orient=tk.HORIZONTAL,
                           label="Frecuencia de corte (Hz)", bg="#000000", fg="white", troughcolor="#3CD31A",
                           highlightthickness=0)
        self.f1.set(4000)
        self.f1.pack()
        self.f1.bind("<ButtonRelease-1>", lambda e: self.update_filter())

        # === Visualización de coeficientes ===
        self.coef_label = tk.Label(root, text="Coeficientes: Ninguno", bg="#000000", fg="white", justify=tk.LEFT, font=("Consolas", 9))
        self.coef_label.pack(pady=(5, 10))

        # === Inicialización ===
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

            coef_text = f"Coeficientes (b): {np.round(b, 4)}\nCoeficientes (a): {np.round(a, 4)}"
            self.coef_label.config(text=coef_text)
            self.status_label.config(text=f"{filtro} actualizado")

    def on_filter_change(self):
        filtro = self.filter_type_var.get()
        if filtro == "Pasa baja configurable":
            self.f1_container.pack(pady=(0, 10))
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
