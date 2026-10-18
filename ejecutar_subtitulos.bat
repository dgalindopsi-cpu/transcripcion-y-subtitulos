@echo off
REM ============================================================
REM Ejecuta los_subtitulos.py usando su propio venv
REM (.venv_subtitulos), sin necesidad de activarlo a mano.
REM
REM Uso:
REM   1) Doble click en este archivo -> te pedira la ruta del audio/video
REM   2) O arrastra un archivo de audio/video encima de este .bat
REM ============================================================

setlocal

REM Carpeta donde esta este .bat (asume que los .py y el venv estan aqui mismo)
set "PROYECTO=%~dp0"
set "VENV_PY=%PROYECTO%.venv_subtitulos\Scripts\python.exe"
set "SCRIPT=%PROYECTO%los_subtitulos.py"

if not exist "%VENV_PY%" (
    echo [X] No se encontro el venv en: %VENV_PY%
    echo     Verifica que .venv_subtitulos esta en la misma carpeta que este .bat.
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [X] No se encontro los_subtitulos.py en: %PROYECTO%
    pause
    exit /b 1
)

REM Si arrastraste un archivo encima del .bat, llega como %1
set "ARCHIVO=%~1"

if "%ARCHIVO%"=="" (
    set /p ARCHIVO="Pega la ruta completa del archivo de audio/video: "
)

echo.
echo Usando Python: %VENV_PY%
echo Archivo: %ARCHIVO%
echo.

"%VENV_PY%" "%SCRIPT%" "%ARCHIVO%"

echo.
echo ============================================================
echo Proceso terminado. Revisa la carpeta del archivo original
echo para ver los resultados (.txt, .srt, .log).
echo ============================================================
pause
