"""
Transcripcion de audio/video con faster-whisper (GPU/CUDA)
------------------------------------------------------------
Uso:
    python los_subtitulos.py "C:\\ruta\\a\\tu\\video.mp4"
    python los_subtitulos.py "C:\\ruta\\a\\tu\\carpeta"

Si le pasas una CARPETA, procesa todos los archivos de audio/video
soportados dentro de ella y de TODAS sus subcarpetas, cargando el
modelo una sola vez para todo el lote. Si un archivo falla, sigue
con los demas y al final muestra un resumen.

Si no pasas la ruta como argumento, el script te la pedira por consola.

Salidas (en la MISMA carpeta que el archivo de entrada, mismo nombre base):
    - .txt  -> transcripcion en texto plano
    - .srt  -> subtitulos con timestamps (si EXPORT_SRT = True)
    - .log  -> registro de progreso / errores, util para diagnosticar cortes

Dependencias (dentro de tu venv):
    pip install faster-whisper av numpy scipy noisereduce
    (av ya viene incluido como dependencia de faster-whisper normalmente)

PREPROCESS_AUDIO controla la limpieza automatica de audio (eco/ganancia/
ruido de fondo), aplicada solo cuando el analisis detecta que hace falta.
"""

import sys
import os
import time
import traceback

# --- Fix para Windows: hacer visibles las DLLs de CUDA (cublas/cudnn) ---
if os.name == "nt":
    try:
        import site
        for base in site.getsitepackages() + [site.getusersitepackages()]:
            for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
                path = os.path.join(base, sub)
                if os.path.isdir(path):
                    os.add_dll_directory(path)
    except Exception as e:
        print(f"[!] No se pudo registrar automaticamente las DLLs de CUDA: {e}")

import av
import numpy as np
from faster_whisper import WhisperModel

import audio_preprocess

# ============ CONFIGURACION ============
MODEL_SIZE = "large-v3"      # opciones: tiny, base, small, medium, large-v2, large-v3
DEVICE = "cuda"               # usa la GPU
COMPUTE_TYPE = "float16"      # si falla por memoria, se reintenta automaticamente con int8_float16
LANGUAGE = "es"               # idioma del audio. Pon None para autodeteccion
BEAM_SIZE = 10
INCLUDE_TIMESTAMPS = False    # True si quieres [00:00:01 -> 00:00:04] antes de cada linea en el .txt
EXPORT_SRT = True             # genera tambien un archivo .srt con subtitulos reales

USE_VAD = True                # si sospechas que el VAD esta cortando el audio, prueba con False
VAD_PARAMETERS = dict(
    min_silence_duration_ms=500,   # menos agresivo detectando "fin de voz" que el default (2000ms)
    speech_pad_ms=200,
)

CONDITION_ON_PREVIOUS_TEXT = False  # False evita bucles/alucinaciones en audios largos

PREPROCESS_AUDIO = True       # limpieza automatica (eco/ganancia/ruido), solo si hace falta por umbral
TARGET_SR = 16000              # sample rate interno para el preprocesamiento
# ========================================


def load_waveform_mono(path, target_sr=TARGET_SR):
    """Decodifica audio/video con PyAV a mono float32 [-1, 1] a target_sr.
    No requiere FFmpeg del sistema (PyAV lo trae incluido)."""
    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=target_sr)

    chunks = []
    for frame in container.decode(stream):
        for rframe in resampler.resample(frame):
            chunks.append(rframe.to_ndarray())
    for rframe in resampler.resample(None):  # flush
        chunks.append(rframe.to_ndarray())
    container.close()

    audio_int16 = np.concatenate(chunks, axis=1)
    waveform = (audio_int16.astype(np.float32) / 32768.0).reshape(-1)
    return waveform


def get_input_path():
    if len(sys.argv) > 1:
        return sys.argv[1].strip('"')
    return input("Pega la ruta completa del archivo o carpeta (video/audio): ").strip('"')


SUPPORTED_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm",
                   ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma"}


def listar_archivos_carpeta(carpeta):
    """Lista archivos de audio/video soportados dentro de una carpeta,
    entrando tambien a TODAS sus subcarpetas (busqueda recursiva).
    Orden: alfabetico por carpeta, y dentro de cada carpeta alfabetico
    por archivo -- resultado deterministico y secuencial."""
    archivos = []
    for raiz, subcarpetas, nombres in os.walk(carpeta):
        subcarpetas.sort()  # controla el orden en que os.walk baja a las subcarpetas
        for nombre in sorted(nombres):
            ruta = os.path.join(raiz, nombre)
            if os.path.splitext(nombre)[1].lower() in SUPPORTED_EXTS:
                archivos.append(ruta)
    return archivos


def format_timestamp_txt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_model(device, compute_type):
    return WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)


def procesar_archivo(input_path, model):
    """Procesa un solo archivo con el modelo ya cargado. Devuelve True/False
    segun si termino bien, sin detener el procesamiento de otros archivos."""
    base, _ = os.path.splitext(input_path)
    output_txt = base + ".txt"
    output_srt = base + ".srt"
    output_log = base + "_transcripcion.log"

    print(f"\n{'='*70}")
    print(f"Archivo: {input_path}")
    print(f"{'='*70}")
    print(f"Salida TXT: {output_txt}")
    if EXPORT_SRT:
        print(f"Salida SRT: {output_srt}")
    print(f"Log: {output_log}")

    log_f = open(output_log, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    t0 = time.time()

    # --- Preprocesamiento de audio (eco/ganancia/ruido), solo si hace falta ---
    transcribe_input = input_path
    if PREPROCESS_AUDIO:
        log("Decodificando y analizando audio (eco/ganancia/ruido)...")
        try:
            mono = load_waveform_mono(input_path)
            transcribe_input = audio_preprocess.preprocess(mono, TARGET_SR, log=log)
        except Exception as e:
            log(f"[!] Fallo el preprocesamiento, se continua con el archivo original: {e}")
            log(traceback.format_exc())
            transcribe_input = input_path
        log("")

    vad_kwargs = {"vad_filter": USE_VAD}
    if USE_VAD:
        vad_kwargs["vad_parameters"] = VAD_PARAMETERS

    try:
        segments, info = model.transcribe(
            transcribe_input,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            **vad_kwargs,
        )
    except Exception as e:
        log(f"[X] Error iniciando la transcripcion: {e}")
        log(traceback.format_exc())
        log_f.close()
        return False

    log(f"Idioma detectado: {info.language} (prob. {info.language_probability:.2f})")
    total_duration = getattr(info, "duration", None)
    if total_duration:
        log(f"Duracion total del audio: {format_timestamp_txt(total_duration)}")
    log("Transcribiendo...\n")

    txt_f = open(output_txt, "w", encoding="utf-8")
    srt_f = open(output_srt, "w", encoding="utf-8") if EXPORT_SRT else None

    idx = 0
    last_end = 0.0
    exito = True
    try:
        for segment in segments:
            idx += 1
            texto = segment.text.strip()
            last_end = segment.end

            pct = f" ({last_end/total_duration*100:.1f}%)" if total_duration else ""
            log(f"[{format_timestamp_txt(segment.start)} -> {format_timestamp_txt(segment.end)}]{pct} {texto}")

            if INCLUDE_TIMESTAMPS:
                linea = f"[{format_timestamp_txt(segment.start)} -> {format_timestamp_txt(segment.end)}] {texto}"
            else:
                linea = texto
            txt_f.write(linea + "\n")
            txt_f.flush()
            os.fsync(txt_f.fileno())

            if EXPORT_SRT:
                srt_f.write(f"{idx}\n")
                srt_f.write(f"{format_timestamp_srt(segment.start)} --> {format_timestamp_srt(segment.end)}\n")
                srt_f.write(f"{texto}\n\n")
                srt_f.flush()
                os.fsync(srt_f.fileno())

    except Exception as e:
        exito = False
        log(f"\n[X] La transcripcion se detuvo con un error en el segmento {idx} "
            f"(ultimo timestamp procesado: {format_timestamp_txt(last_end)}):")
        log(f"    {type(e).__name__}: {e}")
        log(traceback.format_exc())

        if "CUDA" in str(e) or "memory" in str(e).lower():
            log("\n[!] Parece un error de memoria de GPU. Prueba cambiando COMPUTE_TYPE "
                "a 'int8_float16' o 'int8' al inicio del script y vuelve a correrlo.")

    finally:
        txt_f.close()
        if srt_f:
            srt_f.close()
        log_f.close()

    elapsed = time.time() - t0
    print(f"\nListo (o detenido). Revisa {output_log} para ver hasta donde llego exactamente.")
    print(f"TXT guardado en: {output_txt}")
    if EXPORT_SRT:
        print(f"SRT guardado en: {output_srt}")
    print(f"Tiempo: {elapsed/60:.1f} min")
    return exito


def main():
    input_path = get_input_path()

    if os.path.isdir(input_path):
        archivos = listar_archivos_carpeta(input_path)
        if not archivos:
            print(f"[X] No se encontraron archivos de audio/video soportados en la carpeta: {input_path}")
            sys.exit(1)
        print(f"Carpeta detectada: se encontraron {len(archivos)} archivo(s) para procesar.")
    elif os.path.isfile(input_path):
        archivos = [input_path]
    else:
        print(f"[X] No se encontro el archivo o carpeta: {input_path}")
        sys.exit(1)

    print(f"Cargando modelo '{MODEL_SIZE}' en {DEVICE} ({COMPUTE_TYPE})...\n")
    try:
        model = load_model(DEVICE, COMPUTE_TYPE)
    except Exception as e:
        print(f"[X] Error cargando el modelo en {DEVICE}/{COMPUTE_TYPE}: {e}")
        traceback.print_exc()
        sys.exit(1)

    resultados = {"ok": [], "error": []}
    t_total0 = time.time()
    for i, archivo in enumerate(archivos, 1):
        print(f"\n\n### Procesando {i}/{len(archivos)}: {os.path.basename(archivo)} ###")
        try:
            exito = procesar_archivo(archivo, model)
        except Exception as e:
            exito = False
            print(f"[X] Error inesperado procesando {archivo}: {e}")
            traceback.print_exc()
        (resultados["ok"] if exito else resultados["error"]).append(archivo)

    print(f"\n{'='*70}")
    print(f"RESUMEN: {len(resultados['ok'])} exitosos, {len(resultados['error'])} con error, "
          f"de {len(archivos)} archivo(s) totales.")
    print(f"Tiempo total: {(time.time()-t_total0)/60:.1f} min")
    if resultados["error"]:
        print("Archivos con error:")
        for a in resultados["error"]:
            print(f"  - {a}")


if __name__ == "__main__":
    main()