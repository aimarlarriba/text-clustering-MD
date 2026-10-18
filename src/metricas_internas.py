import numpy as np


def calcular_matriz_cofenetica_scratch(matriz_linkage):
    """
    Construye desde cero (Scratch) la matriz de distancias cofenéticas C de tamaño (N x N).
    C[i, j] representa la distancia (altura del dendrograma) a la que las instancias
    originales i y j se unen en un mismo cluster por primera vez.

    Pre-condiciones:
        - matriz_linkage: numpy.ndarray de dimensión (N - 1, 4) donde cada fila es
          [id_cluster_1, id_cluster_2, distancia_fusion, n_muestras].

    Post-condiciones:
        - Devuelve una matriz simétrica C de dimensión (N, N) con diagonal 0.
    """
    L = np.asarray(matriz_linkage, dtype=np.float64)
    N = L.shape[0] + 1
    C = np.zeros((N, N), dtype=np.float64)

    # Inicialmente cada instancia i está en su propio cluster unitario {i}
    miembros = {i: [i] for i in range(N)}

    # Detectamos automáticamente si linkage_matrix usa IDs estándar (N + paso)
    # o si reutiliza los índices de fila (0 .. N-1)
    usa_ids_estandar = np.max(L[:, :2]) >= N

    for paso in range(N - 1):
        c1 = int(L[paso, 0])
        c2 = int(L[paso, 1])
        altura_fusion = L[paso, 2]

        lista_1 = miembros[c1]
        lista_2 = miembros[c2]

        # Registrar la altura de fusión para todos los pares entre ambos clusters
        for u in lista_1:
            for v in lista_2:
                C[u, v] = altura_fusion
                C[v, u] = altura_fusion

        # Actualizar miembros del cluster fusionado
        fusionados = lista_1 + lista_2
        if usa_ids_estandar:
            miembros[N + paso] = fusionados
        else:
            miembros[c1] = fusionados

    return C


def calcular_cpcc_scratch(matriz_distancias_original, matriz_linkage):
    """
    Calcula el Coeficiente de Correlación Cofenética (CPCC - Sokal & Rohlf, 1962)
    desde cero (Scratch). Mide la correlación lineal de Pearson entre las distancias
    originales entre pares y las distancias cofenéticas impuestas por el dendrograma.

    Pre-condiciones:
        - matriz_distancias_original: numpy.ndarray simétrica (N, N) con las distancias originales.
        - matriz_linkage: numpy.ndarray (N - 1, 4) resultante del clustering jerárquico.

    Post-condiciones:
        - Devuelve un float en [-1.0, 1.0]. Valores cercanos a 1 indican que el árbol
          jerárquico preserva fielmente las distancias originales (baja distorsión).
    """
    D = np.asarray(matriz_distancias_original, dtype=np.float64)
    N = D.shape[0]
    if N < 2:
        raise ValueError("Se necesitan al menos 2 instancias para calcular el CPCC.")

    C = calcular_matriz_cofenetica_scratch(matriz_linkage)

    # Extraer únicamente los N*(N-1)/2 pares de la matriz triangular superior (i < j)
    filas, cols = np.triu_indices(N, k=1)
    d_pares = D[filas, cols]
    c_pares = C[filas, cols]

    # Medias aritméticas
    d_media = np.mean(d_pares)
    c_media = np.mean(c_pares)

    # Desviaciones respecto a la media
    d_diff = d_pares - d_media
    c_diff = c_pares - c_media

    numerador = np.sum(d_diff * c_diff)
    denominador = np.sqrt(np.sum(d_diff ** 2) * np.sum(c_diff ** 2))

    if denominador == 0.0:
        return 0.0

    cpcc = numerador / denominador
    return round(float(np.clip(cpcc, -1.0, 1.0)), 4)


def calcular_silhouette_scratch(matriz_distancias, etiquetas_predichas, devolver_detalle=False):
    """
    Calcula el Silhouette Score desde cero (Scratch) comparando para cada muestra i:
      - Cohesión a(i): distancia media de i al resto de muestras de su mismo cluster.
      - Separación b(i): distancia media de i a las muestras del cluster vecino más cercano.
      - s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Pre-condiciones:
        - matriz_distancias: numpy.ndarray simétrica (N, N) con las distancias originales.
        - etiquetas_predichas: array-like 1D de longitud N con los clusters asignados.
        - El número de clusters K debe cumplir 2 <= K <= N - 1.

    Post-condiciones:
        - Si devolver_detalle=False: devuelve el Silhouette Score medio global redondeado a 4 decimales.
        - Si devolver_detalle=True: devuelve un diccionario con el score global, los vectores
          s_i, a_i (cohesión) y b_i (separación).
    """
    D = np.asarray(matriz_distancias, dtype=np.float64)
    etiquetas = np.asarray(etiquetas_predichas)
    N = D.shape[0]

    clusters_unicos = np.unique(etiquetas)
    K = len(clusters_unicos)

    if K < 2 or K >= N:
        raise ValueError(
            f"El Silhouette Score requiere entre 2 y N-1 clusters (actual: K={K}, N={N})."
        )

    a_i = np.zeros(N, dtype=np.float64)
    b_i = np.zeros(N, dtype=np.float64)
    s_i = np.zeros(N, dtype=np.float64)

    indices_por_cluster = {c: np.where(etiquetas == c)[0] for c in clusters_unicos}

    for i in range(N):
        c_propio = etiquetas[i]
        miembros_propios = indices_por_cluster[c_propio]

        # Si el cluster tiene un único punto, por definición s(i) = 0
        if len(miembros_propios) <= 1:
            s_i[i] = 0.0
            continue

        # 1. Cohesión a(i): distancia media a los puntos de su propio cluster (sin contarse a sí mismo)
        suma_propio = np.sum(D[i, miembros_propios]) - D[i, i]
        a_i[i] = suma_propio / (len(miembros_propios) - 1)

        # 2. Separación b(i): mínima distancia media a cualquier otro cluster diferente
        min_dist_vecino = np.inf
        for c_vecino in clusters_unicos:
            if c_vecino == c_propio:
                continue
            miembros_vecino = indices_por_cluster[c_vecino]
            dist_media_vecino = np.mean(D[i, miembros_vecino])
            if dist_media_vecino < min_dist_vecino:
                min_dist_vecino = dist_media_vecino

        b_i[i] = min_dist_vecino

        # 3. Silueta individual s(i)
        max_ab = max(a_i[i], b_i[i])
        s_i[i] = (b_i[i] - a_i[i]) / max_ab if max_ab > 0.0 else 0.0

    score_global = round(float(np.mean(s_i)), 4)

    if devolver_detalle:
        return {
            "Silhouette Score": score_global,
            "Cohesion Media a(i)": round(float(np.mean(a_i)), 4),
            "Separacion Media b(i)": round(float(np.mean(b_i)), 4),
            "s_i": s_i,
            "a_i": a_i,
            "b_i": b_i,
        }

    return score_global