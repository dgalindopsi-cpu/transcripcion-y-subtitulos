@echo off
REM ============================================================
REM Menu de inicio: pregunta que quieres hacer antes de lanzar
REM nada. Un solo .bat para todo el proyecto.
REM ============================================================

setlocal enabledelayedexpansion

set "PROYECTO=%~dp0"

:MENU
cls
echo ============================================================
echo   TRANSCRIPCION / DIARIZACION - Menu de inicio
echo ============================================================
echo.
echo   1) Solo subtitulos (consola, venv_subtitulos)
echo   2) Transcripcion + diarizacion (consola, venv_diarizado)
echo   3) Abrir interfaz grafica (GUI)
echo   4) Salir
echo.
set /p OPCION="Elige una opcion (1-4): "

if "%OPCION%"=="1" goto SUBTITULOS
if "%OPCION%"=="2" goto DIARIZADO
if "%OPCION%"=="3" goto GUI
if "%OPCION%"=="4" goto FIN

echo.
echo Opcion no valida. Intenta de nuevo.
pause
goto MENU

:SUBTITULOS
set "VENV_PY=%PROYECTO%.venv_subtitulos\Scripts\python.exe"
set "SCRIPT=%PROYECTO%los_subtitulos.py"
if not exist "%VENV_PY%" (
    echo.
    echo [X] No se encontro el venv en: %VENV_PY%
    echo     Verifica que .venv_subtitulos esta en esta misma carpeta.
    pause
    goto MENU
)
set /p ARCHIVO="Pega la ruta completa del archivo de audio/video: "
echo.
"%VENV_PY%" "%SCRIPT%" "%ARCHIVO%"
echo.
echo Proceso terminado. Revisa la carpeta del archivo original.
pause
goto MENU

:DIARIZADO
set "VENV_PY=%PROYECTO%.venv_diarizado\Scripts\python.exe"
set "SCRIPT=%PROYECTO%transcribir_diarizado.py"
if not exist "%VENV_PY%" (
    echo.
    echo [X] No se encontro el venv en: %VENV_PY%
    echo     Verifica que .venv_diarizado esta en esta misma carpeta.
    pause
    goto MENU
)
set /p ARCHIVO="Pega la ruta completa del archivo de audio/video: "
echo.
"%VENV_PY%" "%SCRIPT%" "%ARCHIVO%"
echo.
echo Proceso terminado. Revisa la carpeta del archivo original.
pause
goto MENU

:GUI
REM La GUI solo necesita tkinter (incluido en Python base), asi que
REM cualquiera de los dos venvs sirve para abrirla. Se usa el de
REM diarizado por defecto; si no existe, prueba con el de subtitulos.
set "VENV_PY=%PROYECTO%.venv_diarizado\Scripts\python.exe"
if not exist "%VENV_PY%" set "VENV_PY=%PROYECTO%.venv_subtitulos\Scripts\python.exe"
set "SCRIPT=%PROYECTO%gui_transcripcion.py"

if not exist "%VENV_PY%" (
    echo.
    echo [X] No se encontro ningun venv para abrir la GUI.
    pause
    goto MENU
)
if not exist "%SCRIPT%" (
    echo.
    echo [X] No se encontro gui_transcripcion.py en: %PROYECTO%
    pause
    goto MENU
)

echo.
echo Abriendo interfaz grafica...
start "" "%VENV_PY%" "%SCRIPT%"
goto FIN

:FIN
endlocal
