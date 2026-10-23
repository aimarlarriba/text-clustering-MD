# ==============================================================================
# Script de ejecución de pipeline para Windows PowerShell
# ==============================================================================
param (
    [string]$step = "all",
    [int]$n_muestras = 500,
    [switch]$test
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " text-clustering-MD: Ejecutor de Pipeline PowerShell" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$pythonCmd = "python"
if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
    $pythonCmd = "python3"
    if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
        Write-Error "No se encontro un interprete de Python en el PATH."
        exit 1
    }
}

if ($test) {
    Write-Host "[INFO] Ejecutando suite de tests unitarios..." -ForegroundColor Yellow
    & $pythonCmd main.py --test
} else {
    Write-Host "[INFO] Ejecutando pipeline (paso: $step, muestras: $n_muestras)..." -ForegroundColor Yellow
    & $pythonCmd main.py --step $step --n_muestras $n_muestras
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[EXITO] Ejecucion finalizada correctamente." -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] El script finalizo con codigo de error: $LASTEXITCODE" -ForegroundColor Red
}
