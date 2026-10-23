"""
Paquete principal de text-clustering-MD:
Minería de datos y clustering aglomerativo no supervisado sobre posts clínicos de Reddit.

Distribución de Responsabilidades por Paquetes y Módulos:
- Aimar Larriba:
    * main.py: CLI unificado, flags de inferencia y orquestación.
    * src/embeddings.py: Representaciones densas SBERT, métricas de distancia scratch,
      análisis de contraste de Aggarwal y profiling empírico O(N^2).
    * src/experimentos.py: Estudios de dimensionalidad, representaciones y baseline KMeans.
    * src/config.py: Configuración centralizada de rutas y reproducibilidad.
- Adrián Ramírez:
    * src/metricas_internas.py: CPCC, Cohesión SSE, Separación BSS, Silueta y Dunn.
    * src/clustering/lance_williams.py: Coeficientes de actualización recursiva Lance-Williams.
- Urko Horas:
    * src/clustering/hierarchical.py: Motor aglomerativo SAHN iterativo scratch y linkage.
    * toy_example.py: Demostración didáctica paso a paso autocontenida.
- María Fernández:
    * src/evaluacion.py: Validación externa C-SSRS, Holdout sin fuga de etiquetas,
      matriz de confusión heatmap, Codebook semántico 6x3 e inferencia intra-cluster.
    * src/pipeline.py: Ingesta, limpieza y normalización del corpus C-SSRS.
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
