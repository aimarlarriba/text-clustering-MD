"""
Configuración centralizada del proyecto text-clustering-MD.
Define rutas absolutas y relativas reproducibles, hiperparámetros por defecto
y el mapeo de ontología clínica C-SSRS (Gaur et al., 2019).
"""

from pathlib import Path

# --- Rutas del Proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "500_Reddit_users_posts_labels.csv"

# Datos procesados y artefactos
PROCESSED_DATA_DIR = PROJECT_ROOT / "data_processed"
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings_sbert.npy"
LABELS_FILE = PROCESSED_DATA_DIR / "etiquetas_reales.npy"
DIST_COSINE_FILE = PROCESSED_DATA_DIR / "matriz_distancias_cosine.npy"
DIST_EUCLIDEAN_FILE = PROCESSED_DATA_DIR / "matriz_distancias_euclidean.npy"
POSTS_TEXT_FILE = PROCESSED_DATA_DIR / "textos_posts.json"
METRICS_PROFILING_FILE = PROCESSED_DATA_DIR / "metricas_espacios_y_profiling.json"
COMPARISON_PLOT_FILE = PROCESSED_DATA_DIR / "comparativa_distancias.png"

# Informes y figuras
REPORTS_DIR = PROCESSED_DATA_DIR / "graficas_informe"

# --- Hiperparámetros por Defecto ---
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_N_SAMPLES = 500
RANDOM_SEED = 42

# --- Ontología Clínica C-SSRS (Zenodo Gaur et al., 2019) ---
CLASS_MAP = {
    "indicator": 0,
    "ideation": 1,
    "behavior": 2,
    "attempt": 3,
    "supportive": 4,
}

CLASS_NAMES = {
    0: "Indicator",
    1: "Ideation",
    2: "Behavior",
    3: "Attempt",
    4: "Supportive",
}


def ensure_directories():
    """Crea los directorios esenciales del proyecto si aún no existen."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
