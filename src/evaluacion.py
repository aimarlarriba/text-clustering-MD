import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster
from src.metricas_internas import calcular_cpcc_scratch, calcular_silhouette_scratch


def calcular_metricas_externas(etiquetas_reales, etiquetas_predichas):
    """
        Calcula métricas de evaluación externa (Rand Index, Jaccard, Fowlkes-Mallows) desde cero (Scratch).

        Pre-condiciones:
            - etiquetas_reales: array-like 1D de tamaño N con las clases reales (Ground-Truth).
            - etiquetas_predichas: array-like 1D de tamaño N con los clusters asignados por el modelo.
            - len(etiquetas_reales) == len(etiquetas_predichas) > 0.

        Post-condiciones:
            - Devuelve un diccionario con las tres métricas calculadas en el rango [0, 1].
        """
    n = len(etiquetas_reales)
    n11 = n00 = n10 = n01 = 0

    for i in range(n):
        for j in range(i + 1, n):
            mismo_real = (etiquetas_reales[i] == etiquetas_reales[j])
            mismo_pred = (etiquetas_predichas[i] == etiquetas_predichas[j])

            if mismo_real and mismo_pred:
                n11 += 1
            elif not mismo_real and not mismo_pred:
                n00 += 1
            elif not mismo_real and mismo_pred:
                n10 += 1
            elif mismo_real and not mismo_pred:
                n01 += 1

    total_pares = n11 + n00 + n10 + n01
    rand_index = (n11 + n00) / total_pares if total_pares > 0 else 0
    jaccard = n11 / (n11 + n10 + n01) if (n11 + n10 + n01) > 0 else 0

    denominador_fm = np.sqrt((n11 + n10) * (n11 + n01))
    fowlkes_mallows = n11 / denominador_fm if denominador_fm > 0 else 0

    return {"Rand Index": round(rand_index, 4), "Jaccard": round(jaccard, 4),
            "Fowlkes-Mallows": round(fowlkes_mallows, 4)}


def cortar_arbol(matriz_linkage, altura_h):
    clusters_asignados = fcluster(matriz_linkage, t=altura_h, criterion='distance')
    return clusters_asignados


def visualizar_pca_clusters(vectores_densos, etiquetas_predichas):
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
    Asigna un nuevo post a un cluster y muestra por pantalla los textos más similares.

    Pre-condiciones:
        - vector_nuevo: numpy array 1D de tamaño D (dimensión del embedding).
        - vectores_entrenamiento: numpy matrix de tamaño (N, D).
        - etiquetas_entrenamiento: numpy array 1D de tamaño N.
        - textos_entrenamiento: lista de strings de tamaño N con los posts originales.
        - k: entero > 0, número de textos similares a mostrar.

    Post-condiciones:
        - Imprime los K textos más cercanos y sus distancias.
        - Devuelve un int/string con la etiqueta del cluster asignado.
    """
    # 1. Calcular la distancia euclídea entre el nuevo post y todos los de entrenamiento
    distancias = np.linalg.norm(vectores_entrenamiento - vector_nuevo, axis=1)

    # 2. Obtener los índices de los 'k' posts con menor distancia (los más cercanos)
    indices_cercanos = np.argsort(distancias)[:k]

    # 3. Asignar el cluster basado en el vecino más cercano (1-NN)
    cluster_asignado = etiquetas_entrenamiento[indices_cercanos[0]]

    # 4. Mostrar los textos originales por consola
    print(f"\n--- NUEVO POST ASIGNADO AL CLUSTER: {cluster_asignado} ---")
    print(f"Los {k} posts del corpus más semánticamente parecidos son:")
    for i, idx in enumerate(indices_cercanos):
        print(f"  {i + 1}. [Distancia: {distancias[idx]:.4f}] -> \"{textos_entrenamiento[idx]}\"")

    return cluster_asignado


# --- BLOQUE DE EJECUCIÓN CON DATOS REALES ---
if __name__ == "__main__":
    import os
    import json
    import numpy as np

    print("Iniciando evaluación con datos reales...")

    # 1. CARGAR LOS DATOS PROCESADOS POR AIMAR
    # Buscamos la carpeta 'data_processed' que crea el pipeline de Aimar
    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta_datos = os.path.join(carpeta_raiz, "data_processed")

    ruta_embeddings = os.path.join(carpeta_datos, "embeddings_sbert.npy")
    ruta_etiquetas = os.path.join(carpeta_datos, "etiquetas_reales.npy")
    ruta_textos = os.path.join(carpeta_datos, "textos_posts.json")

    try:
        print("Cargando embeddings, textos y etiquetas guardados...")
        vectores_densos = np.load(ruta_embeddings)
        etiquetas_reales = np.load(ruta_etiquetas)

        with open(ruta_textos, "r", encoding="utf-8") as f:
            textos_reales = json.load(f)

        print(f"¡Éxito! {len(textos_reales)} posts cargados correctamente.")
    except FileNotFoundError:
        print(
            "Error: No se encuentran los datos. Dile a Aimar que ejecute su 'pipeline_aimar.py' primero o asegúrate de haber hecho un Git Pull.")
        exit()

    # 2. GENERAR EL CLUSTERING (Parte de Urko)
    # import sys
    # sys.path.append(carpeta_raiz)
    # from src.clustering.hierarchical import hacer_clustering

    # print("Calculando matriz de enlaces (Linkage)...")
    # matriz_linkage = hacer_clustering(vectores_densos)

    # 3. TU PARTE: CORTAR EL ÁRBOL Y EVALUAR
    # print("Cortando dendrograma...")
    # altura_h = 2.0  # <- Ajustaréis este valor al ver el dendrograma

    # predicciones = cortar_arbol(matriz_linkage, altura_h)

    # resultados = calcular_metricas_externas(etiquetas_reales, predicciones)
    # print("Métricas finales:", resultados)

    # print("Abriendo gráfico PCA real...")
    # visualizar_pca_clusters(vectores_densos, predicciones)

    # print("Probando inferencia con el primer texto del corpus...")
    # inferir_y_mostrar_cercanos(vectores_densos[0], vectores_densos, predicciones, textos_reales, k=3)