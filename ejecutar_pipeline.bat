@echo off
REM ==============================================================================
REM  Script de ejecución de pipeline para Windows (CMD / Batch)
REM ==============================================================================
echo [INFO] Iniciando pipeline de Text Clustering...

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] No se encontro Python en el PATH. Por favor instala Python 3.10+.
    pause
    exit /b 1
)

python main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] El pipeline finalizo con errores.
    exit /b %ERRORLEVEL%
)

echo [OK] Pipeline completado con exito.
