"""
Módulo común del equipo para la Práctica 2 (Clasificación Supervisada).
Garantiza que todos los integrantes trabajen exactamente con:
  - La misma partición: 70% Train (350) / 15% Dev (75) / 15% Test (75)
  - La misma semilla (RANDOM_SEED = 42)
  - El mismo corte estratificado (stratify = y)
  - El mismo Baseline (Clustering Jerárquico sobre Embeddings 384D)
"""

import os
import ast
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering

RANDOM_SEED = 42
RUTA_CSV_DEFECTO = "data/500_Reddit_users_posts_labels.csv"
RUTA_CACHE_384D = "data/embeddings_384d_comun.npy"


def limpiar_texto_reddit(val):
    """Convierte listas de posts serializadas como string ['post1', 'post2'] en un texto plano."""
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            lista = ast.literal_eval(val_str)
            if isinstance(lista, list):
                return " ".join(str(x) for x in lista)
        except Exception:
            pass
    return val_str


def cargar_particion_oficial_equipo(ruta_csv=RUTA_CSV_DEFECTO, ruta_cache_emb=RUTA_CACHE_384D, seed=RANDOM_SEED):
    """
    Precondición: Existe el archivo CSV en `ruta_csv`.
    Postcondición: Devuelve un diccionario con los textos, los embeddings 384D y las etiquetas `y`
    exactamente divididos en 70% Train (350), 15% Dev (75) y 15% Test (75) con `stratify=y` y `seed=42`.
    """
    df = pd.read_csv(ruta_csv)
    textos = df["Post"].fillna("").apply(limpiar_texto_reddit).values
    y = df["Label"].astype(str).values
    usuarios = df["User"].astype(str).values

    # Cargar o generar los Embeddings 384D comunes para todo el equipo
    if os.path.exists(ruta_cache_emb):
        X_384d = np.load(ruta_cache_emb)
    else:
        from sentence_transformers import SentenceTransformer
        print("[*] Calculando Embeddings 384D comunes (all-MiniLM-L6-v2)...")
        modelo_st = SentenceTransformer("all-MiniLM-L6-v2")
        X_384d = modelo_st.encode(textos.tolist(), show_progress_bar=True)
        os.makedirs(os.path.dirname(ruta_cache_emb), exist_ok=True)
        np.save(ruta_cache_emb, X_384d)

    indices = np.arange(len(y))

    # 1º Corte: 70% Train (350 muestras) y 30% Temporal (150 muestras)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y, test_size=0.30, random_state=seed, stratify=y
    )

    # 2º Corte: Dividir el 30% Temporal en dos mitades iguales -> 15% Dev (75) y 15% Test (75)
    idx_dev, idx_test, y_dev, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    return {
        "usuarios": (usuarios[idx_train], usuarios[idx_dev], usuarios[idx_test]),
        "textos": (textos[idx_train], textos[idx_dev], textos[idx_test]),
        "X_384": (X_384d[idx_train], X_384d[idx_dev], X_384d[idx_test]),
        "y": (y_train, y_dev, y_test),
        "idx": (idx_train, idx_dev, idx_test),
        "clases": np.unique(y)
    }


class BaselineClusteringJerarquico384D:
    """
    Baseline común del equipo:
    Agrupa los embeddings 384D de Train mediante Clustering Jerárquico,
    asigna a cada clúster la clase mayoritaria en Train, y predice nuevas muestras
    (Dev/Test) asignándolas al centroide del clúster más cercano.
    """
    def __init__(self, n_clusters=5, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.centroides = None
        self.mapa_clase = {}
        self.probabilidades_cluster = None
        self.clases_ = None

    def fit(self, X_train, y_train):
        self.clases_ = np.unique(y_train)
        hc = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        clusters_tr = hc.fit_predict(X_train)

        self.centroides = np.zeros((self.n_clusters, X_train.shape[1]))
        self.probabilidades_cluster = np.zeros((self.n_clusters, len(self.clases_)))

        for c in range(self.n_clusters):
            mask = (clusters_tr == c)
            self.centroides[c] = X_train[mask].mean(axis=0)
            y_c = y_train[mask]
            for idx_cls, cls_name in enumerate(self.clases_):
                self.probabilidades_cluster[c, idx_cls] = np.sum(y_c == cls_name) / len(y_c)
            self.mapa_clase[c] = self.clases_[np.argmax(self.probabilidades_cluster[c])]
        return self

    def obtener_cluster_id(self, X):
        """Útil para Aimar: devuelve el ID del clúster más cercano para usar como feature."""
        dist = cdist(X, self.centroides, metric="euclidean")
        return np.argmin(dist, axis=1)

    def predict(self, X):
        clusters = self.obtener_cluster_id(X)
        return np.array([self.mapa_clase[c] for c in clusters])

    def predict_proba(self, X):
        clusters = self.obtener_cluster_id(X)
        return self.probabilidades_cluster[clusters]


if __name__ == "__main__":
    datos = cargar_particion_oficial_equipo()
    y_tr, y_dev, y_te = datos["y"]
    print("=== PARTICIÓN OFICIAL DEL EQUIPO 70/15/15 (SEED = 42, STRATIFY = Y) ===")
    print(f"Total Train (70%): {len(y_tr)} | Total Dev (15%): {len(y_dev)} | Total Test (15%): {len(y_te)}")
    df_dist = pd.DataFrame({
        "Train (70%)": pd.Series(y_tr).value_counts(),
        "Dev (15%)": pd.Series(y_dev).value_counts(),
        "Test (15%)": pd.Series(y_te).value_counts()
    })
    print("\nDistribución de clases por partición:")
    print(df_dist)
