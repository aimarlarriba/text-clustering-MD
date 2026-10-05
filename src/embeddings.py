import os
import numpy as np
from sentence_transformers import SentenceTransformer


def generar_embeddings(textos, nombre_modelo="all-MiniLM-L6-v2", normalizar=True):
    """Genera representaciones densas contextuales X in R^(N x 384) con Sentence-BERT."""
    modelo = SentenceTransformer(nombre_modelo)
    textos = [str(t) if t is not None else "" for t in textos]
    vectores = modelo.encode(textos, normalize_embeddings=normalizar)
    return vectores.astype("float32")


def guardar_embeddings(X: np.ndarray, ruta_archivo: str) -> None:
    """Guarda la matriz de embeddings en formato binario de NumPy (.npy)."""
    directorio = os.path.dirname(os.path.abspath(ruta_archivo))
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    np.save(ruta_archivo, X)
    print(f"[OK] Embeddings guardados en: {ruta_archivo} (Forma: {X.shape})")


def cargar_embeddings(ruta_archivo: str) -> np.ndarray:
    """Carga una matriz de embeddings previamente guardada desde un archivo .npy."""
    X = np.load(ruta_archivo)
    print(f"[OK] Embeddings cargados desde: {ruta_archivo} (Forma: {X.shape})")
    return X

