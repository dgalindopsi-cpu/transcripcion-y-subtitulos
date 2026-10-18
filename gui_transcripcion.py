"""
gui_transcripcion.py
------------------------------------------------------------
Interfaz grafica simple (Tkinter, sin dependencias extra) para
correr los_subtitulos.py o transcribir_diarizado.py sin usar la
consola. Muestra el progreso en vivo (mismo log que ya imprimen
ambos scripts) y deja elegir el archivo con un dialogo nativo.

No reemplaza la logica de los scripts: los ejecuta como subproceso
(python los_subtitulos.py "archivo") y muestra su salida en pantalla,
asi que cualquier cambio que hagas en esos scripts se refleja aqui
automaticamente sin tocar la GUI.

Uso:
    python gui_transcripcion.py

Requiere: Python con tkinter (viene incluido en la instalacion
estandar de Python en Windows).
"""

import os
import sys
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_SUBTITULOS = os.path.join(SCRIPT_DIR, "los_subtitulos.py")
SCRIPT_DIARIZADO = os.path.join(SCRIPT_DIR, "transcribir_diarizado.py")

# ============ RUTAS DE CADA VENV ============
# Como usas DOS entornos virtuales distintos (uno para diarizado, otro
# para subtitulos con Python global), indica aqui el ejecutable de
# python.exe de cada uno. Si dejas None, se usa el mismo Python con el
# que abriste esta GUI (sirve solo si ambos scripts comparten entorno).
#
# Ejemplo de ruta tipica de un venv en Windows:
#   r"C:\Transcripcion\.venv_diarizado\Scripts\python.exe"
#
# Si los_subtitulos.py corre en tu Python global, deja PYTHON_SUBTITULOS
# en None (usara "python" del PATH del sistema).
PYTHON_DIARIZADO = None   # ej: r"C:\Transcripcion\.venv\Scripts\python.exe"
PYTHON_SUBTITULOS = None  # ej: None si usas Python global, o ruta a otro venv
# ================================================


def _resolver_python(ruta_configurada):
    """Devuelve el ejecutable de Python a usar: la ruta configurada si
    existe, o 'python' del sistema/PATH como fallback (para Python global),
    o el mismo Python que corre la GUI como ultimo recurso."""
    if ruta_configurada and os.path.isfile(ruta_configurada):
        return ruta_configurada
    if ruta_configurada:
        # se configuro una ruta pero no existe -> avisar en vez de fallar silencioso
        return None
    return "python"


class TranscripcionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Transcripcion / Diarizacion")
        self.root.geometry("760x560")
        self.root.minsize(600, 420)

        self.input_path = tk.StringVar()
        self.modo = tk.StringVar(value="subtitulos")  # "subtitulos" o "diarizado"
        self.proceso = None
        self.log_queue = queue.Queue()
        self.corriendo = False

        self._build_ui()
        self._poll_log_queue()

    # ---------- UI ----------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm_top = ttk.Frame(self.root)
        frm_top.pack(fill="x", **pad)

        ttk.Label(frm_top, text="Archivo o carpeta de audio/video:").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(frm_top, textvariable=self.input_path)
        entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        frm_top.columnconfigure(0, weight=1)

        frm_botones_archivo = ttk.Frame(frm_top)
        frm_botones_archivo.grid(row=1, column=1)
        ttk.Button(frm_botones_archivo, text="Elegir archivo...", command=self._elegir_archivo).pack(side="left")
        ttk.Button(frm_botones_archivo, text="Elegir carpeta...", command=self._elegir_carpeta).pack(side="left", padx=(6, 0))

        self.lbl_info_carpeta = ttk.Label(frm_top, foreground="#666")
        self.lbl_info_carpeta.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        frm_modo = ttk.LabelFrame(self.root, text="Modo")
        frm_modo.pack(fill="x", **pad)
        ttk.Radiobutton(
            frm_modo, text="Solo subtitulos (transcripcion simple)",
            variable=self.modo, value="subtitulos",
        ).pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(
            frm_modo, text="Transcripcion + diarizacion (identifica quien habla)",
            variable=self.modo, value="diarizado",
        ).pack(anchor="w", padx=10, pady=2)

        frm_botones = ttk.Frame(self.root)
        frm_botones.pack(fill="x", **pad)
        self.btn_iniciar = ttk.Button(frm_botones, text="Iniciar", command=self._iniciar)
        self.btn_iniciar.pack(side="left")
        self.btn_cancelar = ttk.Button(frm_botones, text="Cancelar", command=self._cancelar, state="disabled")
        self.btn_cancelar.pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(0, 6))

        frm_log = ttk.LabelFrame(self.root, text="Progreso")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt_log = tk.Text(frm_log, wrap="word", state="disabled", bg="#111", fg="#ddd")
        self.txt_log.pack(fill="both", expand=True, side="left")
        scroll = ttk.Scrollbar(frm_log, command=self.txt_log.yview)
        scroll.pack(side="right", fill="y")
        self.txt_log.configure(yscrollcommand=scroll.set)

        self.status_var = tk.StringVar(value="Listo.")
        ttk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(0, 8))

    # ---------- acciones ----------
    SUPPORTED_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm",
                       ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}

    def _elegir_archivo(self):
        path = filedialog.askopenfilename(
            title="Selecciona el archivo de audio/video",
            filetypes=[
                ("Audio/Video", "*.mp4 *.mkv *.mov *.avi *.wav *.mp3 *.m4a *.flac *.ogg"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.input_path.set(path)
            self.lbl_info_carpeta.configure(text="")

    def _elegir_carpeta(self):
        path = filedialog.askdirectory(title="Selecciona la carpeta con audios/videos")
        if not path:
            return
        self.input_path.set(path)
        # cuenta cuantos archivos soportados hay (recursivo) para dar feedback inmediato
        total = 0
        for _raiz, _subs, nombres in os.walk(path):
            total += sum(1 for n in nombres if os.path.splitext(n)[1].lower() in self.SUPPORTED_EXTS)
        if total == 0:
            self.lbl_info_carpeta.configure(
                text="[!] No se encontraron archivos de audio/video soportados en esta carpeta ni en sus subcarpetas.",
                foreground="#c33",
            )
        else:
            self.lbl_info_carpeta.configure(
                text=f"Carpeta seleccionada: {total} archivo(s) encontrado(s) (incluyendo subcarpetas).",
                foreground="#666",
            )

    def _iniciar(self):
        path = self.input_path.get().strip()
        if not path:
            messagebox.showwarning("Falta el archivo", "Elige un archivo o carpeta de audio/video primero.")
            return
        if not os.path.isfile(path) and not os.path.isdir(path):
            messagebox.showerror("No encontrado", f"No se encontro el archivo ni la carpeta:\n{path}")
            return
        if self.corriendo:
            return

        script = SCRIPT_DIARIZADO if self.modo.get() == "diarizado" else SCRIPT_SUBTITULOS
        ruta_configurada = PYTHON_DIARIZADO if self.modo.get() == "diarizado" else PYTHON_SUBTITULOS
        python_exe = _resolver_python(ruta_configurada)
        if python_exe is None:
            messagebox.showerror(
                "Ruta de Python invalida",
                f"La ruta configurada para este modo no existe:\n{ruta_configurada}\n\n"
                "Revisa PYTHON_DIARIZADO / PYTHON_SUBTITULOS al inicio de gui_transcripcion.py.",
            )
            return

        self._limpiar_log()
        self._append_log(f"Ejecutando: {os.path.basename(script)}\nEntrada: {path}\nPython: {python_exe}\n{'-'*60}\n")
        self.corriendo = True
        self.btn_iniciar.configure(state="disabled")
        self.btn_cancelar.configure(state="normal")
        self.progress.start(12)
        self.status_var.set("Procesando... esto puede tardar varios minutos segun la duracion del archivo.")

        hilo = threading.Thread(target=self._correr_proceso, args=(script, path, python_exe), daemon=True)
        hilo.start()

    def _correr_proceso(self, script, path, python_exe):
        try:
            self.proceso = subprocess.Popen(
                [python_exe, script, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=SCRIPT_DIR,
            )
            for line in self.proceso.stdout:
                self.log_queue.put(line)
            self.proceso.wait()
            codigo = self.proceso.returncode
        except Exception as e:
            self.log_queue.put(f"\n[X] Error lanzando el proceso: {e}\n")
            codigo = -1

        self.log_queue.put(("__DONE__", codigo))

    # ---------- log / polling ----------
    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__DONE__":
                    self._on_finished(item[1])
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _on_finished(self, codigo):
        self.corriendo = False
        self.proceso = None
        self.progress.stop()
        self.btn_iniciar.configure(state="normal")
        self.btn_cancelar.configure(state="disabled")
        if codigo == 0:
            self.status_var.set("Listo. Revisa la carpeta del archivo original para ver los resultados.")
            self._append_log(f"\n{'-'*60}\nProceso terminado correctamente.\n")
        else:
            self.status_var.set(f"Termino con errores (codigo {codigo}). Revisa el log arriba.")
            self._append_log(f"\n{'-'*60}\nProceso termino con codigo {codigo}.\n")

    def _cancelar(self):
        if self.proceso and self.proceso.poll() is None:
            self.proceso.terminate()
            self._append_log("\n[!] Cancelado por el usuario.\n")
        self.corriendo = False
        self.progress.stop()
        self.btn_iniciar.configure(state="normal")
        self.btn_cancelar.configure(state="disabled")
        self.status_var.set("Cancelado.")

    def _append_log(self, text):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", text)
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _limpiar_log(self):
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    TranscripcionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
