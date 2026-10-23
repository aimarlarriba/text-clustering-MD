"""
Paquete principal de text-clustering-MD:
Minería de datos y clustering aglomerativo no supervisado sobre posts clínicos de Reddit.
"""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    CLASS_MAP,
    CLASS_NAMES,
)

__version__ = "1.0.0"
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_FILE",
    "PROCESSED_DATA_DIR",
    "REPORTS_DIR",
    "CLASS_MAP",
    "CLASS_NAMES",
]
