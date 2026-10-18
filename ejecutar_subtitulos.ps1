# ============================================================
# Ejecuta los_subtitulos.py usando su propio venv
# (.venv_subtitulos), sin necesidad de activarlo a mano.
#
# Uso:
#   .\ejecutar_subtitulos.ps1
#   .\ejecutar_subtitulos.ps1 "C:\ruta\a\tu\archivo.mp4"
#
# Si Windows bloquea la ejecucion de scripts .ps1, corre una vez:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# ============================================================

param(
    [string]$Archivo
)

$Proyecto = $PSScriptRoot
$VenvPy = Join-Path $Proyecto ".venv_subtitulos\Scripts\python.exe"
$Script = Join-Path $Proyecto "los_subtitulos.py"

if (-not (Test-Path $VenvPy)) {
    Write-Host "[X] No se encontro el venv en: $VenvPy" -ForegroundColor Red
    Write-Host "    Verifica que .venv_subtitulos esta en la misma carpeta que este script."
    Read-Host "Presiona Enter para salir"
    exit 1
}

if (-not (Test-Path $Script)) {
    Write-Host "[X] No se encontro los_subtitulos.py en: $Proyecto" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

if (-not $Archivo) {
    $Archivo = Read-Host "Pega la ruta completa del archivo de audio/video"
}

Write-Host ""
Write-Host "Usando Python: $VenvPy"
Write-Host "Archivo: $Archivo"
Write-Host ""

& $VenvPy $Script $Archivo

Write-Host ""
Write-Host "============================================================"
Write-Host "Proceso terminado. Revisa la carpeta del archivo original"
Write-Host "para ver los resultados (.txt, .srt, .log)."
Write-Host "============================================================"
Read-Host "Presiona Enter para cerrar"
