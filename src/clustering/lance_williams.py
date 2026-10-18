import numpy as np


def obtener_parametros_lance_williams(tipo_enlace, n_i, n_j, n_k=1):
    """
    Devuelve los coeficientes (alpha_i, alpha_j, beta, gamma) de la fórmula
    recursiva de Lance-Williams según el criterio de enlace.

    Fórmula de Lance-Williams (1967):
    d(C_i U C_j, C_k) = alpha_i * d(C_i, C_k) + alpha_j * d(C_j, C_k)
                        + beta * d(C_i, C_j) + gamma * |d(C_i, C_k) - d(C_j, C_k)|

    Pre-condiciones:
        - tipo_enlace: str en {'single', 'complete', 'average'}.
        - n_i, n_j: enteros >= 1 con el número de elementos en los clusters C_i y C_j.
        - n_k: entero >= 1 con el número de elementos en el cluster C_k.

    Post-condiciones:
        - Devuelve una tupla de 4 floats: (alpha_i, alpha_j, beta, gamma).
    """
    enlace = tipo_enlace.lower()

    if enlace == "single":
        # Enlace simple (vecino más cercano): min(d_ik, d_jk)
        return 0.5, 0.5, 0.0, -0.5

    elif enlace == "complete":
        # Enlace completo (vecino más lejano): max(d_ik, d_jk)
        return 0.5, 0.5, 0.0, 0.5

    elif enlace in ["average", "upgma"]:
        # Enlace promedio (UPGMA): ponderación continua proporcional al tamaño de cada cluster
        total_ij = float(n_i + n_j)
        alpha_i = n_i / total_ij
        alpha_j = n_j / total_ij
        return alpha_i, alpha_j, 0.0, 0.0

    else:
        raise ValueError(f"Tipo de enlace '{tipo_enlace}' no soportado en Lance-Williams.")


def actualizar_matriz_lance_williams(matriz_distancias, cluster1, cluster2, tipo_enlace, tamanos_clusters):
    """
    Actualiza in-place la matriz de distancias tras fusionar cluster1 (C_i) y cluster2 (C_j)
    aplicando la fórmula recursiva de Lance-Williams.

    El nuevo cluster fusionado (C_i U C_j) se almacena en la fila/columna 'cluster1',
    mientras que la fila/columna 'cluster2' se desactiva asignando infinito (np.inf).

    Pre-condiciones:
        - matriz_distancias: numpy.ndarray cuadrada (N x N) de distancias.
        - cluster1, cluster2: índices enteros (0 <= cluster1, cluster2 < N) a fusionar.
        - tipo_enlace: 'single', 'complete' o 'average'.
        - tamanos_clusters: diccionario o lista con el tamaño actual de cada índice activo.

    Post-condiciones:
        - Actualiza la fila y columna 'cluster1' con las nuevas distancias d(C_i U C_j, C_k).
        - Anula la fila y columna 'cluster2' con np.inf para evitar que vuelva a seleccionarse.
        - Actualiza tamanos_clusters[cluster1] = |C_i| + |C_j|.
    """
    N = matriz_distancias.shape[0]
    n_i = tamanos_clusters[cluster1]
    n_j = tamanos_clusters[cluster2]
    dist_ij = matriz_distancias[cluster1, cluster2]

    for k in range(N):
        if k == cluster1 or k == cluster2:
            continue

        d_ik = matriz_distancias[cluster1, k]
        d_jk = matriz_distancias[cluster2, k]

        # Si el cluster k ya fue absorbido previamente (distancia infinita), lo saltamos
        if np.isinf(d_ik) or np.isinf(d_jk):
            continue

        n_k = tamanos_clusters[k]
        alpha_i, alpha_j, beta, gamma = obtener_parametros_lance_williams(
            tipo_enlace, n_i, n_j, n_k
        )

        nueva_distancia = (
            alpha_i * d_ik
            + alpha_j * d_jk
            + beta * dist_ij
            + gamma * abs(d_ik - d_jk)
        )

        matriz_distancias[cluster1, k] = nueva_distancia
        matriz_distancias[k, cluster1] = nueva_distancia

    # La diagonal principal debe mantenerse en infinito para no fusionar un cluster consigo mismo
    matriz_distancias[cluster1, cluster1] = np.inf

    # Desactivar el cluster2 (absorbido dentro de cluster1)
    matriz_distancias[cluster2, :] = np.inf
    matriz_distancias[:, cluster2] = np.inf

    # Actualizar el tamaño acumulado del nuevo cluster en la posición cluster1
    tamanos_clusters[cluster1] = n_i + n_j