"""
Módulo de algoritmos de clustering jerárquico aglomerativo (SAHN).
Implementación desde cero basada en la fórmula recursiva de Lance-Williams (1967).
"""

from .lance_williams import (
    obtener_parametros_lance_williams,
    actualizar_matriz_lance_williams,
)
from .hierarchical import MotorJerarquico

__all__ = [
    "obtener_parametros_lance_williams",
    "actualizar_matriz_lance_williams",
    "MotorJerarquico",
]
