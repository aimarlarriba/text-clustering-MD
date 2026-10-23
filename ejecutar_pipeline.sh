#!/usr/bin/env bash
# ==============================================================================
# Script de ejecución de pipeline para Linux / macOS / Bash
# ==============================================================================
set -e

echo "======================================================================"
echo " text-clustering-MD: Ejecutor de Pipeline Bash"
echo "======================================================================"

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

$PYTHON_CMD main.py "$@"
