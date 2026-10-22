import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster


def calcular_metricas_externas(labels_true, labels_pred):
    """
    Calcula las métricas de validación externa evaluando los pares de instancias.

    Pre-condiciones:
    - labels_true y labels_pred deben ser arrays 1D de igual longitud (N,).

    Post-condiciones:
    - Devuelve un diccionario con las tres métricas (Rand, Jaccard, Fowlkes-Mallows).
    - Todos los índices devueltos están acotados en [0.0, 1.0].
    """
    labels_true = np.asarray(labels_true).flatten()
    labels_pred = np.asarray(labels_pred).flatten()

    assert len(labels_true) == len(labels_pred), "Las dimensiones de las etiquetas no coinciden"

    n = len(labels_true)

    mat_true = (labels_true[:, None] == labels_true[None, :])
    mat_pred = (labels_pred[:, None] == labels_pred[None, :])

    upper_tri_indices = np.triu_indices(n, k=1)
    pairs_true = mat_true[upper_tri_indices]
    pairs_pred = mat_pred[upper_tri_indices]

    a = np.sum(pairs_true & pairs_pred)
    b = np.sum(~pairs_true & ~pairs_pred)
    c = np.sum(~pairs_true & pairs_pred)
    d = np.sum(pairs_true & ~pairs_pred)

    rand_index = (a + b) / (a + b + c + d) if (a + b + c + d) > 0 else 0.0
    jaccard = a / (a + c + d) if (a + c + d) > 0 else 0.0
    fowlkes_mallows = a / np.sqrt((a + c) * (a + d)) if (a + c) > 0 and (a + d) > 0 else 0.0

    return {
        "rand_index": float(rand_index),
        "jaccard": float(jaccard),
        "fowlkes_mallows": float(fowlkes_mallows)
    }


def cortar_arbol(matriz_linkage, altura_h):
    """
    Genera los clústeres planos a partir del dendrograma jerárquico.

    Pre-condiciones:
    - matriz_linkage: matriz numpy de tamaño (N-1, 4).
    - altura_h: valor float positivo.

    Post-condiciones:
    - Devuelve un array 1D con las etiquetas de clúster.
    """
    return fcluster(matriz_linkage, t=altura_h, criterion='distance')


def visualizar_pca_clusters(vectores_densos, etiquetas_predichas):
    """
    Reduce la dimensionalidad del espacio vectorial y lo grafica en 2D.

    Pre-condiciones:
    - vectores_densos: matriz numpy de tamaño (N, D).
    - etiquetas_predichas: array 1D de tamaño N con los IDs de los clústeres.

    Post-condiciones:
    - Muestra un gráfico de dispersión con la proyección PCA.
    """
    pca = PCA(n_components=2)
    vectores_2d = pca.fit_transform(vectores_densos)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(vectores_2d[:, 0], vectores_2d[:, 1], c=etiquetas_predichas, cmap='viridis', alpha=0.8,
                          edgecolor='k')
    plt.title("Proyección PCA de los Clusters")
    plt.colorbar(scatter, label="Cluster ID")
    plt.show()


def inferir_y_mostrar_cercanos(vector_nuevo, vectores_entrenamiento, etiquetas_entrenamiento, textos_entrenamiento,
                               k=3):
    """
    Asigna un nuevo post a un cluster usando 1-NN.

    Pre-condiciones:
    - vector_nuevo: array 1D de tamaño D.
    - vectores_entrenamiento: matriz de tamaño (N, D).
    - etiquetas_entrenamiento: array 1D de tamaño N.
    - textos_entrenamiento: lista de strings de tamaño N.
    - k: entero > 0.

    Post-condiciones:
    - Imprime los K textos más cercanos.
    - Devuelve la etiqueta del cluster asignado.
    """
    distancias = np.linalg.norm(vectores_entrenamiento - vector_nuevo, axis=1)
    indices_cercanos = np.argsort(distancias)[:k]

    cluster_asignado = etiquetas_entrenamiento[indices_cercanos[0]]

    print(f"\n--- NUEVO POST ASIGNADO AL CLUSTER: {cluster_asignado} ---")
    print(f"Los {k} posts más parecidos son:")
    for i, idx in enumerate(indices_cercanos):
        print(f"  {i + 1}. [Distancia: {distancias[idx]:.4f}] -> \"{textos_entrenamiento[idx]}\"")

    return cluster_asignado


if __name__ == "__main__":
    print("Iniciando evaluación...")

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta_datos = os.path.join(carpeta_raiz, "data_processed")
    sys.path.append(carpeta_raiz)

    # 1. Cargar los datos procesados
    try:
        vectores_densos = np.load(os.path.join(carpeta_datos, "embeddings_sbert.npy"))
        etiquetas_reales = np.load(os.path.join(carpeta_datos, "etiquetas_reales.npy"))
        matriz_distancias = np.load(os.path.join(carpeta_datos, "matriz_distancias_cosine.npy"))

        with open(os.path.join(carpeta_datos, "textos_posts.json"), "r", encoding="utf-8") as f:
            textos_reales = json.load(f)
    except FileNotFoundError:
        print("Error: No se encuentran los datos. Ejecuta pipeline.py primero.")
        exit()

    # 2. Generar el clustering
    from src.clustering.hierarchical import MotorJerarquico

    motor = MotorJerarquico(tipo_enlace="average")
    matriz_linkage = motor.fit(matriz_distancias)

    # 3. Cortar el dendrograma
    altura_h = 0.70
    predicciones = cortar_arbol(matriz_linkage, altura_h)

    print("\n--- RESULTADOS DE VALIDACIÓN ---")
    resultados = calcular_metricas_externas(etiquetas_reales, predicciones)
    print(f"Rand Index: {resultados.get('rand_index', 0):.4f}")
    print(f"Jaccard: {resultados.get('jaccard', 0):.4f}")
    print(f"Fowlkes-Mallows: {resultados.get('fowlkes_mallows', 0):.4f}")

    visualizar_pca_clusters(vectores_densos, predicciones)

    inferir_y_mostrar_cercanos(vectores_densos[0], vectores_densos, predicciones, textos_reales, k=3)