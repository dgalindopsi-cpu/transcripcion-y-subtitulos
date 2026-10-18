# Transcripción y subtítulos de audio

Herramientas para Windows que generan transcripciones TXT y subtítulos SRT con `faster-whisper`. Incluyen procesamiento por lotes, limpieza de audio opcional, interfaz gráfica y una opción de diarización para distinguir hablantes.

## Instalación

1. Instala Python y crea los entornos usados por los lanzadores: `py -m venv .venv_subtitulos` y `py -m venv .venv_diarizado`.
2. Instala las dependencias básicas en ambos entornos: `python -m pip install -r requirements.txt`, usando el `python.exe` de cada entorno.
3. Para diarización, instala además `pyannote.audio` y PyTorch compatible con tu GPU en `.venv_diarizado`.
4. Ejecuta `iniciar.bat` y elige la modalidad.

La configuración actual usa CUDA y `large-v3`; requiere una GPU NVIDIA con memoria suficiente. Para diarización, acepta los modelos de pyannote en Hugging Face y configura la variable de entorno `HF_TOKEN`. Los archivos de salida se guardan junto al audio original.
