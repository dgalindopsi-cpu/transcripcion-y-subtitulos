"""
audio_preprocess.py
------------------------------------------------------------
Modulo compartido de preprocesamiento de audio para las grabaciones
de salon de clases: mide el estado del audio (nivel, ruido, "eco")
y aplica solo las correcciones que hacen falta, por umbral.

Se usa tanto desde transcribir_diarizado.py como desde
los_subtitulos.py (y desde la GUI que envuelve a ambos).

Requiere (en el venv):
    pip install noisereduce scipy numpy

No requiere FFmpeg del sistema: trabaja sobre arrays numpy que ya
vienen decodificados (por PyAV o por donde sea).
"""

import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

# ============ UMBRALES (ajustables) ============
# Ganancia: si el RMS esta por debajo de este nivel en dBFS, se sube.
TARGET_DBFS = -20.0
GAIN_THRESHOLD_DBFS = -26.0   # si el audio ya esta mas fuerte que esto, no se toca
MAX_GAIN_DB = 18.0            # limite para no disparar el ruido de fondo al maximo

# Ruido de fondo: si el "piso de ruido" estimado supera este dBFS
# relativo al pico, se aplica noisereduce.
NOISE_FLOOR_THRESHOLD_DB = -35.0

# Reverb/eco: proxy simple basado en la energia que "sobrevive" tras
# los picos (cola de decaimiento). Si supera este ratio, se aplica
# el filtro de-reverb.
REVERB_RATIO_THRESHOLD = 0.35

# Clipping: si el pico absoluto de la señal supera este umbral,
# se considera que el audio esta saturado/distorsionado en el origen.
CLIPPING_THRESHOLD = 0.99
# ==================================================


def detect_clipping(waveform: np.ndarray, threshold: float = CLIPPING_THRESHOLD) -> dict:
    """Detecta si la señal esta saturada (clipping) en la fuente.
    Esto NO se puede arreglar despues de grabado -- solo se informa,
    para que sepas que hay que bajar el volumen de entrada la proxima vez."""
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    clipped_ratio = float(np.mean(np.abs(waveform) >= threshold)) if waveform.size else 0.0
    return {
        "peak": peak,
        "is_clipped": peak >= threshold,
        "clipped_sample_ratio": clipped_ratio,
    }


def _rms_dbfs(waveform: np.ndarray) -> float:
    """RMS en dBFS de una señal float32 en rango [-1, 1]."""
    rms = np.sqrt(np.mean(np.square(waveform), dtype=np.float64)) + 1e-12
    return 20 * np.log10(rms)


def _estimate_noise_floor_dbfs(waveform: np.ndarray, sr: int) -> float:
    """Estima el piso de ruido tomando el percentil 10 de energia
    por ventanas de 50ms (ventanas 'silenciosas' = ruido de fondo)."""
    win = int(0.05 * sr)
    if win < 1 or waveform.size < win:
        return _rms_dbfs(waveform)
    n_wins = waveform.size // win
    trimmed = waveform[: n_wins * win].reshape(n_wins, win)
    energies = np.sqrt(np.mean(np.square(trimmed), axis=1, dtype=np.float64)) + 1e-12
    floor = np.percentile(energies, 10)
    return 20 * np.log10(floor)


def _estimate_reverb_ratio(waveform: np.ndarray, sr: int) -> float:
    """Proxy barato de 'cuanta cola de energia' queda tras los picos
    de la señal, como indicador de reverberacion/eco. No es un
    medidor de RT60 real, pero es suficiente para decidir el umbral."""
    win = int(0.02 * sr)  # ventanas de 20ms
    if win < 1 or waveform.size < win * 10:
        return 0.0
    n_wins = waveform.size // win
    trimmed = waveform[: n_wins * win].reshape(n_wins, win)
    envelope = np.sqrt(np.mean(np.square(trimmed), axis=1, dtype=np.float64)) + 1e-12

    peak_thresh = np.percentile(envelope, 90)
    peak_idxs = np.where(envelope >= peak_thresh)[0]
    if peak_idxs.size == 0:
        return 0.0

    # para cada pico, mira la energia 3-6 ventanas despues (60-120ms)
    # vs la energia del propio pico -> si sigue alta, hay cola/eco
    ratios = []
    for idx in peak_idxs:
        tail_start, tail_end = idx + 3, idx + 6
        if tail_end >= envelope.size:
            continue
        tail_energy = envelope[tail_start:tail_end].mean()
        ratios.append(tail_energy / envelope[idx])
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def analyze(waveform: np.ndarray, sr: int) -> dict:
    """Analiza la señal y devuelve metricas + que correcciones hacen falta.

    waveform: array float32 mono en rango [-1, 1], shape (n_samples,)
    """
    level_dbfs = _rms_dbfs(waveform)
    noise_floor_dbfs = _estimate_noise_floor_dbfs(waveform, sr)
    reverb_ratio = _estimate_reverb_ratio(waveform, sr)
    clipping = detect_clipping(waveform)

    needs_gain = level_dbfs < GAIN_THRESHOLD_DBFS
    needs_denoise = noise_floor_dbfs > NOISE_FLOOR_THRESHOLD_DB
    needs_dereverb = reverb_ratio > REVERB_RATIO_THRESHOLD

    return {
        "level_dbfs": level_dbfs,
        "noise_floor_dbfs": noise_floor_dbfs,
        "reverb_ratio": reverb_ratio,
        "needs_gain": needs_gain,
        "needs_denoise": needs_denoise,
        "needs_dereverb": needs_dereverb,
        "clipping": clipping,
    }


def apply_gain(waveform: np.ndarray, level_dbfs: float) -> np.ndarray:
    """Sube la ganancia hasta TARGET_DBFS, limitado por MAX_GAIN_DB,
    con un limitador simple (clip suave) para evitar distorsion dura."""
    gain_db = min(TARGET_DBFS - level_dbfs, MAX_GAIN_DB)
    if gain_db <= 0:
        return waveform
    gain_factor = 10 ** (gain_db / 20)
    boosted = waveform * gain_factor
    # limitador suave (soft clip) en vez de recorte duro
    return np.tanh(boosted)


def apply_denoise(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Reduccion de ruido estacionario (ventiladores, zumbido, murmullo
    de fondo constante) via noisereduce."""
    try:
        import noisereduce as nr
    except ImportError as e:
        raise ImportError(
            "Falta 'noisereduce'. Instala con: pip install noisereduce"
        ) from e
    return nr.reduce_noise(y=waveform, sr=sr, stationary=False, prop_decrease=0.85)


def apply_dereverb(waveform: np.ndarray, sr: int) -> np.ndarray:
    """De-reverb ligero: high-pass para quitar retumbe grave de sala +
    un filtro de 'deconvolucion' aproximado (spectral gating suave
    sobre la cola). No sustituye a WPE, pero ayuda notablemente para
    voz en salones de clase sin anadir dependencias pesadas."""
    # 1) high-pass a 100Hz: el retumbe/eco de sala vive mucho en graves
    sos = butter(4, 100, btype="highpass", fs=sr, output="sos")
    filtered = sosfilt(sos, waveform)

    # 2) atenuar la "cola" de reverb con un kernel de decaimiento corto
    #    (deconvolucion aproximada tipo "early reflection suppression")
    kernel_len = int(0.06 * sr)  # 60ms
    if kernel_len > 1:
        decay = np.exp(-np.linspace(0, 5, kernel_len))
        decay /= decay.sum()
        smooth_env = fftconvolve(np.abs(filtered), decay, mode="same")
        # resta una fraccion de la envolvente suavizada (cola) a la señal
        gate = np.clip(1.0 - 0.3 * (smooth_env / (np.abs(filtered).max() + 1e-9)), 0.4, 1.0)
        filtered = filtered * gate

    return filtered.astype(np.float32)


def preprocess(waveform: np.ndarray, sr: int, log=print) -> np.ndarray:
    """Punto de entrada principal: analiza y aplica solo lo necesario.

    waveform: array float32 mono [-1, 1], shape (n_samples,)
    sr: sample rate
    log: funcion de logging (por defecto print)

    Devuelve el waveform procesado (float32, mismo shape).
    """
    metrics = analyze(waveform, sr)
    log(
        f"      Analisis: nivel={metrics['level_dbfs']:.1f} dBFS | "
        f"ruido={metrics['noise_floor_dbfs']:.1f} dBFS | "
        f"reverb_ratio={metrics['reverb_ratio']:.2f}"
    )

    if metrics["clipping"]["is_clipped"]:
        log(
            f"      [!] ADVERTENCIA: clipping detectado en la grabacion original "
            f"(pico={metrics['clipping']['peak']:.3f}, "
            f"{metrics['clipping']['clipped_sample_ratio']*100:.2f}% de muestras saturadas). "
            f"Esto no se puede corregir despues -- baja el volumen de entrada en la proxima grabacion."
        )
    else:
        log(f"      Sin clipping (pico={metrics['clipping']['peak']:.3f})")

    out = waveform

    if metrics["needs_dereverb"]:
        log("      -> Aplicando de-reverb (eco de sala detectado)")
        out = apply_dereverb(out, sr)
    else:
        log("      -> De-reverb no necesario (por debajo del umbral)")

    if metrics["needs_denoise"]:
        log("      -> Aplicando reduccion de ruido de fondo")
        out = apply_denoise(out, sr)
    else:
        log("      -> Reduccion de ruido no necesaria (por debajo del umbral)")

    if metrics["needs_gain"] and not metrics["clipping"]["is_clipped"]:
        # recalcula el nivel tras denoise/dereverb, que pueden bajarlo un poco
        current_level = _rms_dbfs(out)
        log(f"      -> Aplicando ganancia (nivel bajo: {current_level:.1f} dBFS)")
        out = apply_gain(out, current_level)
    elif metrics["needs_gain"]:
        log("      -> Ganancia omitida: subir volumen empeoraria el clipping ya presente")
    else:
        log("      -> Ganancia no necesaria (nivel ya adecuado)")

    return out.astype(np.float32)
