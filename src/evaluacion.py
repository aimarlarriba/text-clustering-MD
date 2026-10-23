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
    - labels_true y labels_pred: arrays 1D de NumPy o listas de igual longitud (N,).

    Post-condiciones:
    - Devuelve un diccionario con Rand Index, Jaccard y Fowlkes-Mallows.
    - Todos los valores numéricos devueltos son floats acotados en [0.0, 1.0].
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


def calcular_metricas_clasificacion_clustering_scratch(y_true, pred_clusters, mapeo_entrenamiento=None):
    """
    Asigna clases reales a los clusters y calcula métricas de clasificación ponderadas.

    Pre-condiciones:
    - y_true y pred_clusters: arrays 1D numéricos de tamaño (N,).
    - mapeo_entrenamiento: opcional, diccionario con la asignación cluster -> clase.

    Post-condiciones:
    - Devuelve un diccionario con el mapeo utilizado, la matriz de confusión (2D array) y
      las métricas ponderadas (accuracy, w_precision, w_recall, w_f1) acotadas en [0.0, 1.0].
    """
    clases_reales = np.unique(y_true)

    if mapeo_entrenamiento is None:
        mapeo = {}
        for c in np.unique(pred_clusters):
            indices_c = np.where(pred_clusters == c)[0]
            etiquetas_c = y_true[indices_c]
            mapeo[c] = np.bincount(etiquetas_c).argmax() if len(etiquetas_c) > 0 else 0
    else:
        mapeo = mapeo_entrenamiento

    y_pred_mapeado = np.array([mapeo.get(c, -1) for c in pred_clusters])
    max_clase = max(clases_reales) if len(clases_reales) > 0 else 0
    matriz_conf = np.zeros((max_clase + 1, max_clase + 1), dtype=int)

    for true_l, pred_l in zip(y_true, y_pred_mapeado):
        if pred_l != -1:
            matriz_conf[true_l, pred_l] += 1

    precisions, recalls, f1s, soporte = {}, {}, {}, {}
    total_instancias = len(y_true)

    for cls in clases_reales:
        tp = matriz_conf[cls, cls]
        fp = np.sum(matriz_conf[:, cls]) - tp
        fn = np.sum(matriz_conf[cls, :]) - tp
        sop = np.sum(matriz_conf[cls, :])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions[cls], recalls[cls], f1s[cls], soporte[cls] = prec, rec, f1, sop

    w_precision = sum(
        precisions[c] * soporte[c] for c in clases_reales) / total_instancias if total_instancias > 0 else 0.0
    w_recall = sum(recalls[c] * soporte[c] for c in clases_reales) / total_instancias if total_instancias > 0 else 0.0
    w_f1 = sum(f1s[c] * soporte[c] for c in clases_reales) / total_instancias if total_instancias > 0 else 0.0
    accuracy = np.trace(matriz_conf) / total_instancias if total_instancias > 0 else 0.0

    return {
        "mapeo_cluster_a_clase": mapeo,
        "matriz_confusion": matriz_conf,
        "accuracy": float(accuracy),
        "w_precision": float(w_precision),
        "w_recall": float(w_recall),
        "w_f1": float(w_f1)
    }


def guardar_matriz_confusion_heatmap(matriz_conf, ruta_salida="data_processed/graficas_informe/heatmap_confusion.png"):
    """
    Exporta la matriz de confusión como un heatmap.

    Pre-condiciones:
    - matriz_conf: array 2D de NumPy con valores enteros no negativos.
    - ruta_salida: string indicando un path válido del sistema de archivos.

    Post-condiciones:
    - Crea los directorios necesarios y guarda una imagen PNG en la ruta_salida.
    - No devuelve ningún valor (retorna None).
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(matriz_conf, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión (Class-to-Cluster)')
    plt.colorbar(label='Número de instancias')

    thresh = matriz_conf.max() / 2.
    for i in range(matriz_conf.shape[0]):
        for j in range(matriz_conf.shape[1]):
            plt.text(j, i, format(matriz_conf[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if matriz_conf[i, j] > thresh else "black")

    plt.ylabel('Clase Real (Ground Truth)')
    plt.xlabel('Clase Asignada (Predicción de Cluster)')
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def cortar_arbol(matriz_linkage, altura_h):
    """
    Genera los clústeres planos a partir del dendrograma jerárquico.

    Pre-condiciones:
    - matriz_linkage: matriz de SciPy de tamaño (N-1, 4) generada por el algoritmo de clustering.
    - altura_h: valor float positivo que determina el umbral de corte.

    Post-condiciones:
    - Devuelve un array 1D de tamaño (N,) con las etiquetas enteras de clúster asignadas.
    """
    return fcluster(matriz_linkage, t=altura_h, criterion='distance')


def visualizar_pca_clusters(vectores_densos, etiquetas_predichas):
    """
    Reduce la dimensionalidad del espacio vectorial y lo grafica en 2D.

    Pre-condiciones:
    - vectores_densos: matriz de NumPy de tamaño (N, D).
    - etiquetas_predichas: array 1D de tamaño (N,) con los IDs de los clústeres.

    Post-condiciones:
    - Muestra un gráfico de dispersión (scatter plot) por pantalla.
    - No devuelve ningún valor (retorna None).
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
    Asigna un nuevo post a un cluster y busca los vecinos más cercanos estrictamente dentro de ese cluster.

    Pre-condiciones:
    - vector_nuevo: array 1D numérico de dimensionalidad D.
    - vectores_entrenamiento: matriz de tamaño (N, D).
    - etiquetas_entrenamiento: array 1D de tamaño N.
    - textos_entrenamiento: lista de strings de tamaño N.
    - k: entero positivo indicando el número de vecinos a mostrar.

    Post-condiciones:
    - Imprime por consola los K textos más cercanos y sus distancias matemáticas.
    - Devuelve el ID numérico del cluster al que se ha asignado la nueva muestra.
    """
    distancias_globales = np.linalg.norm(vectores_entrenamiento - vector_nuevo, axis=1)
    cluster_asignado = etiquetas_entrenamiento[np.argmin(distancias_globales)]

    indices_cluster = np.where(etiquetas_entrenamiento == cluster_asignado)[0]
    vectores_filtrados = vectores_entrenamiento[indices_cluster]
    textos_filtrados = [textos_entrenamiento[i] for i in indices_cluster]

    distancias_locales = np.linalg.norm(vectores_filtrados - vector_nuevo, axis=1)
    indices_cercanos = np.argsort(distancias_locales)[:k]

    print(f"\n--- NUEVO POST ASIGNADO AL CLUSTER: {cluster_asignado} ---")
    print(f"Los {k} posts MÁS CERCANOS DENTRO DEL CLUSTER {cluster_asignado} son:")
    for i, idx in enumerate(indices_cercanos):
        print(f"  {i + 1}. [Distancia: {distancias_locales[idx]:.4f}] -> \"{textos_filtrados[idx]}\"")

    return cluster_asignado


if __name__ == "__main__":
    print("Iniciando evaluación...")
    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta_datos = os.path.join(carpeta_raiz, "data_processed")
    sys.path.append(carpeta_raiz)

    try:
        vectores_densos = np.load(os.path.join(carpeta_datos, "embeddings_sbert.npy"))
        etiquetas_reales = np.load(os.path.join(carpeta_datos, "etiquetas_reales.npy"))
        matriz_distancias = np.load(os.path.join(carpeta_datos, "matriz_distancias_cosine.npy"))
        with open(os.path.join(carpeta_datos, "textos_posts.json"), "r", encoding="utf-8") as f:
            textos_reales = json.load(f)
    except FileNotFoundError:
        print("Error: No se encuentran los datos. Ejecuta pipeline.py primero.")
        exit()

    from src.clustering.hierarchical import MotorJerarquico

    motor = MotorJerarquico(tipo_enlace="average")
    matriz_linkage = motor.fit(matriz_distancias)

    altura_h = 0.70
    predicciones = cortar_arbol(matriz_linkage, altura_h)

    print("\n--- RESULTADOS DE VALIDACIÓN EXTERNA (PARES) ---")
    resultados = calcular_metricas_externas(etiquetas_reales, predicciones)
    print(f"Rand Index: {resultados.get('rand_index', 0):.4f}")
    print(f"Jaccard: {resultados.get('jaccard', 0):.4f}")
    print(f"Fowlkes-Mallows: {resultados.get('fowlkes_mallows', 0):.4f}")

    print("\n--- RESULTADOS DE CLASIFICACIÓN (CLASE A CLUSTER) ---")
    clf_stats = calcular_metricas_clasificacion_clustering_scratch(etiquetas_reales, predicciones)
    print(f"Accuracy: {clf_stats['accuracy']:.4f}")
    print(f"Weighted Precision: {clf_stats['w_precision']:.4f}")
    print(f"Weighted Recall: {clf_stats['w_recall']:.4f}")
    print(f"Weighted F1-Score: {clf_stats['w_f1']:.4f}")

    guardar_matriz_confusion_heatmap(clf_stats["matriz_confusion"])
    print("-> Matriz de confusión guardada en 'data_processed/graficas_informe/heatmap_confusion.png'")

    visualizar_pca_clusters(vectores_densos, predicciones)
    inferir_y_mostrar_cercanos(vectores_densos[0], vectores_densos, predicciones, textos_reales, k=3)