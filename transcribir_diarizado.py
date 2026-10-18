"""
Transcripcion + diarizacion automatica (v3 - sin muestra de voz de referencia)
------------------------------------------------------------
Combina faster-whisper (transcripcion) con pyannote.audio (diarizacion).
Detecta hablantes distintos automaticamente y los etiqueta como
Hablante 1, Hablante 2, etc. segun orden de aparicion.
No requiere ninguna muestra de voz previa.

CAMBIO respecto a la version anterior: en vez de dejar que pyannote lea
el archivo de audio/video por su cuenta (lo cual requiere 'torchcodec' +
FFmpeg instalado en el sistema, que en Windows suele dar problemas de DLLs),
este script carga el audio el mismo usando 'av' (PyAV, que ya viene con
faster-whisper y trae FFmpeg incluido) y se lo pasa a pyannote ya
decodificado en memoria. Funciona igual con archivos de audio o video
(mp4, wav, mp3, etc.) sin necesitar instalar nada adicional.

REQUISITOS PREVIOS (una sola vez):
    1. Cuenta gratuita en https://huggingface.co/join
    2. Acepta las condiciones de estos dos modelos (con la misma cuenta):
       - https://huggingface.co/pyannote/speaker-diarization-community-1
       - https://huggingface.co/pyannote/embedding
    3. Genera un token en https://hf.co/settings/tokens (permiso "Read" basta)
    4. Instala dependencias (dentro de tu venv):
       pip install faster-whisper torch torchvision torchaudio pyannote.audio scipy av noisereduce

CONFIGURA ABAJO:
    HF_TOKEN              -> tu token de Hugging Face (mejor via variable de entorno HF_TOKEN)
    PREPROCESS_AUDIO       -> True para activar limpieza automatica (eco/ganancia/ruido)
    MIN_SPEAKERS / MAX_SPEAKERS -> rango esperado de hablantes en el audio

Uso (con el venv activado):
    python transcribir_diarizado.py "C:\\ruta\\a\\entrevista.mp4"
    python transcribir_diarizado.py "C:\\ruta\\a\\carpeta"

Si le pasas una CARPETA, procesa todos los archivos de audio/video
soportados dentro de ella y de TODAS sus subcarpetas. El pipeline de
diarizacion se carga una sola vez para todo el lote; el modelo de whisper se carga y libera por
archivo (para no exceder la VRAM disponible en GPUs con poca memoria).
Si un archivo falla, se continua con los demas y al final se muestra
un resumen.
"""

import sys
import os
import time
import traceback

import audio_preprocess

# --- Fix para Windows: hacer visibles las DLLs de CUDA ---
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
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
from scipy.spatial.distance import cosine

# ============ CONFIGURACION ============
HF_TOKEN = os.environ.get("HF_TOKEN", "")

PREPROCESS_AUDIO = True   # limpieza automatica (eco/ganancia/ruido)

MODEL_SIZE = "medium"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
LANGUAGE = "es"
BEAM_SIZE = 10
CONDITION_ON_PREVIOUS_TEXT = False

DIARIZATION_PIPELINE = "pyannote/speaker-diarization-community-1"
MIN_SPEAKERS = 2
MAX_SPEAKERS = None

TARGET_SR = 16000

# ===== POSTPROCESAMIENTO DE TURNS =====
# 1. Absorber segmentos cortos al vecino dominante
MIN_TURN_DURATION = 1.5        # segundos; segmentos mas cortos se reasignan

# 2. Fusionar turnos del mismo hablante separados por silencio breve
MAX_GAP_TO_MERGE = 0.5         # segundos; gap <= esto entre mismo hablante -> se fusionan

# 3. Clustering: fusionar etiquetas que pyannote partio del mismo hablante real
CLUSTER_SPEAKERS = True        # True = usar embeddings para reagrupar
CLUSTER_THRESHOLD = 0.20       # distancia coseno; < esto = misma persona (0.15 estricto, 0.25 permisivo)
# ========================================


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


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_waveform(path, target_sr=TARGET_SR):
    """Decodifica audio/video con PyAV (sin depender de torchcodec/FFmpeg del sistema)
    y devuelve el formato dict que pyannote.audio acepta directamente en memoria."""
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
    waveform = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
    return {"waveform": waveform, "sample_rate": target_sr}


def procesar_archivo(input_path, dia_pipeline, torch_device):
    """Procesa un solo archivo (transcripcion + diarizacion + mapeo de voz)
    usando los modelos de diarizacion/embeddings ya cargados (compartidos
    entre todos los archivos del lote). El modelo de whisper se carga y
    libera por archivo para no exceder la VRAM disponible.
    Devuelve True/False segun si termino bien."""

    base, _ = os.path.splitext(input_path)
    output_txt = base + "_diarizado.txt"
    output_srt = base + "_diarizado.srt"
    output_log = base + "_diarizacion.log"

    log_f = open(output_log, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    t0 = time.time()

    # --- 1. Cargar audio en memoria con PyAV y preprocesarlo (eco/ganancia/ruido) ---
    log("[1/5] Decodificando audio con PyAV...")
    try:
        audio_dict = load_waveform(input_path)
        log(f"      Audio principal: {audio_dict['waveform'].shape[-1]/TARGET_SR:.1f}s")
    except Exception as e:
        log(f"[X] Error decodificando el audio: {e}")
        log(traceback.format_exc())
        log_f.close()
        return False

    if PREPROCESS_AUDIO:
        log("\n      Analizando y limpiando audio principal (solo si hace falta)...")
        try:
            mono = audio_dict["waveform"].numpy().reshape(-1).astype(np.float32)
            mono_clean = audio_preprocess.preprocess(mono, TARGET_SR, log=log)
            audio_dict = {"waveform": torch.from_numpy(mono_clean).unsqueeze(0), "sample_rate": TARGET_SR}
        except Exception as e:
            log(f"      [!] Fallo el preprocesamiento, se continua con el audio original: {e}")
            log(traceback.format_exc())
    else:
        log("      Preprocesamiento de audio desactivado (PREPROCESS_AUDIO = False).")

    # el array numpy 1D que usara whisper (mismo audio, ya limpio si aplica)
    whisper_input = audio_dict["waveform"].numpy().reshape(-1).astype(np.float32)

    # --- 2. Transcripcion con faster-whisper (se carga y libera por archivo, por VRAM limitada) ---
    log(f"\n[2/5] Cargando faster-whisper '{MODEL_SIZE}'...")
    try:
        asr_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        seg_gen, info = asr_model.transcribe(
            whisper_input,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            vad_filter=True,
        )
        whisper_segments = []
        log("      Transcribiendo...")
        for seg in seg_gen:
            whisper_segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        log(f"      {len(whisper_segments)} segmentos transcritos.")
    except Exception as e:
        log(f"[X] Error en la transcripcion: {e}")
        log(traceback.format_exc())
        log_f.close()
        return False
    finally:
        if "asr_model" in dir():
            del asr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 3. Diarizacion con pyannote (pipeline ya cargado, compartido entre archivos) ---
    log(f"\n[3/5] Diarizando con: {DIARIZATION_PIPELINE} ...")
    try:
        dia_kwargs = {}
        if MIN_SPEAKERS:
            dia_kwargs["min_speakers"] = MIN_SPEAKERS
        if MAX_SPEAKERS:
            dia_kwargs["max_speakers"] = MAX_SPEAKERS

        log("      Diarizando (separando quien habla cuando)...")
        diarization = dia_pipeline(audio_dict, **dia_kwargs)
    except Exception as e:
        log(f"[X] Error en la diarizacion: {e}")
        log(traceback.format_exc())
        log_f.close()
        return False

    turns = [{"start": t.start, "end": t.end, "speaker": spk}
             for t, spk in diarization.speaker_diarization]
    speakers_found = sorted(set(t["speaker"] for t in turns))
    log(f"      Hablantes detectados: {len(speakers_found)} -> {speakers_found}")

    # --- 4. Postprocesamiento de turns: mejorar calidad de la diarización ---
    log("\n[4/5] Postprocesando diarizacion...")

    # 4a. Absorber segmentos muy cortos al vecino con más overlap temporal
    def absorb_short_turns(turns, min_dur=MIN_TURN_DURATION):
        result = []
        for t in turns:
            dur = t["end"] - t["start"]
            if dur >= min_dur or not result:
                result.append(dict(t))
            else:
                # Buscar vecino más cercano en tiempo y asignar su hablante
                prev_gap = result[-1]["end"] - t["start"] if result else float("inf")
                result.append({**t, "speaker": result[-1]["speaker"]})
        log(f"      [4a] Segmentos cortos absorbidos: "
            f"{len(turns) - len([t for t in turns if t['end']-t['start'] >= min_dur])} turnos < {min_dur}s reasignados")
        return result

    # 4b. Fusionar turnos consecutivos del mismo hablante con gap pequeño
    def merge_same_speaker_gaps(turns, max_gap=MAX_GAP_TO_MERGE):
        if not turns:
            return turns
        merged = [dict(turns[0])]
        for t in turns[1:]:
            last = merged[-1]
            gap = t["start"] - last["end"]
            if t["speaker"] == last["speaker"] and gap <= max_gap:
                last["end"] = t["end"]  # extender el turno anterior
            else:
                merged.append(dict(t))
        log(f"      [4b] Turnos fusionados por gap <= {max_gap}s: {len(turns) - len(merged)} fusiones")
        return merged

    # 4c. Clustering de hablantes: detectar cuando pyannote partió una voz en dos etiquetas
    def cluster_speakers(turns, audio_dict, emb_inf, threshold=CLUSTER_THRESHOLD):
        speakers = sorted(set(t["speaker"] for t in turns))
        if len(speakers) <= 1:
            return turns

        # Calcular embedding representativo de cada hablante (el turno más largo)
        best_turn = {}
        for t in turns:
            dur = t["end"] - t["start"]
            if t["speaker"] not in best_turn or dur > best_turn[t["speaker"]]["dur"]:
                best_turn[t["speaker"]] = {**t, "dur": dur}

        embeddings = {}
        for spk, info_t in best_turn.items():
            try:
                seg = Segment(info_t["start"], info_t["end"])
                embeddings[spk] = emb_inf.crop(audio_dict, seg)
            except Exception as e:
                log(f"      [4c] No se pudo calcular embedding de {spk}: {e}")

        # Construir mapa de fusión: si distancia coseno < threshold -> misma persona
        remap = {spk: spk for spk in speakers}
        merged_any = False
        spk_list = [s for s in speakers if s in embeddings]
        for i in range(len(spk_list)):
            for j in range(i + 1, len(spk_list)):
                a, b = spk_list[i], spk_list[j]
                # Resolver a la raíz actual (por si ya fue remapeado)
                root_a = remap[a]
                root_b = remap[b]
                if root_a == root_b:
                    continue
                dist = cosine(embeddings[a], embeddings[b])
                if dist < threshold:
                    # Fusionar b -> a (la etiqueta menor se queda)
                    for k in remap:
                        if remap[k] == root_b:
                            remap[k] = root_a
                    log(f"      [4c] {b} fusionado con {a} (distancia coseno: {dist:.3f})")
                    merged_any = True

        if not merged_any:
            log(f"      [4c] Ninguna etiqueta fusionada (todas distancias > {threshold})")

        # Aplicar remap a los turns
        return [{**t, "speaker": remap[t["speaker"]]} for t in turns]

    # Aplicar pipeline de postprocesamiento
    turns = absorb_short_turns(turns)
    turns = merge_same_speaker_gaps(turns)

    if CLUSTER_SPEAKERS:
        log("      [4c] Calculando embeddings para clustering...")
        try:
            emb_model = Model.from_pretrained("pyannote/embedding", token=HF_TOKEN)
            emb_inf = Inference(emb_model, window="whole")
            turns = cluster_speakers(turns, audio_dict, emb_inf)
            del emb_model, emb_inf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log(f"      [4c] Clustering omitido: {e}")

    # Renumerar hablantes finales en orden de aparicion
    speaker_names = {}
    counter = 1
    for t in turns:
        if t["speaker"] not in speaker_names:
            speaker_names[t["speaker"]] = f"Hablante {counter}"
            log(f"      {t['speaker']} -> Hablante {counter}")
            counter += 1

    speakers_final = sorted(set(speaker_names.values()))
    log(f"      Hablantes finales: {len(speakers_final)} -> {speakers_final}")

    # --- 5. Fusionar transcripcion + diarizacion por solapamiento de tiempos ---
    log("\n[5/5] Combinando transcripcion con hablantes y guardando resultados...")

    def speaker_for(seg_start, seg_end):
        best_spk, best_overlap = None, 0.0
        for t in turns:
            overlap = min(seg_end, t["end"]) - max(seg_start, t["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = t["speaker"]
        return speaker_names.get(best_spk, "Hablante ?")

    txt_f = open(output_txt, "w", encoding="utf-8")
    srt_f = open(output_srt, "w", encoding="utf-8")
    idx = 0
    for seg in whisper_segments:
        idx += 1
        spk_name = speaker_for(seg["start"], seg["end"])
        line = f"{spk_name}: {seg['text']}"

        txt_f.write(line + "\n")
        txt_f.flush()
        srt_f.write(f"{idx}\n{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}\n{line}\n\n")
        srt_f.flush()

        log(f"  [{fmt_ts(seg['start'])} -> {fmt_ts(seg['end'])}] {line}")

    txt_f.close()
    srt_f.close()

    elapsed = time.time() - t0
    log(f"\nListo. Tiempo: {elapsed/60:.1f} min")
    log(f"TXT: {output_txt}")
    log(f"SRT: {output_srt}")
    log_f.close()
    return True


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

    if not HF_TOKEN:
        print("[X] Falta configurar la variable de entorno HF_TOKEN con tu token de Hugging Face.")
        sys.exit(1)

    torch_device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # --- Cargar UNA VEZ para todo el lote: pipeline de diarizacion + embedding de referencia ---
    print(f"Cargando pipeline de diarizacion: {DIARIZATION_PIPELINE} ...")
    try:
        dia_pipeline = Pipeline.from_pretrained(DIARIZATION_PIPELINE, token=HF_TOKEN)
        dia_pipeline.to(torch_device)
    except Exception as e:
        print(f"[X] Error cargando el pipeline de diarizacion: {e}")
        traceback.print_exc()
        sys.exit(1)

    resultados = {"ok": [], "error": []}
    t_total0 = time.time()
    for i, archivo in enumerate(archivos, 1):
        print(f"\n\n### Procesando {i}/{len(archivos)}: {os.path.basename(archivo)} ###")
        try:
            exito = procesar_archivo(archivo, dia_pipeline, torch_device)
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