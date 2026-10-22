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



def cortar_dendrograma_scratch(matriz_linkage, n_clusters=None, altura_h=None):
    """
    Corta el dendrograma jerárquico desde cero (Scratch) sin usar scipy.cluster.hierarchy.fcluster.
    Permite cortar especificando el número deseado de clusters (n_clusters) o la altura de corte (altura_h).

    Pre-condiciones:
        - matriz_linkage: numpy.ndarray de dimensión (N - 1, 4).
        - Debe especificarse al menos uno de: n_clusters (int en [1, N]) o altura_h (float >= 0).

    Post-condiciones:
        - Devuelve un numpy.ndarray 1D de longitud N con etiquetas enteras consecutivas [0 .. K-1].
    """
    if n_clusters is None and altura_h is None:
        raise ValueError("Debe especificarse 'n_clusters' o 'altura_h' para cortar el dendrograma.")

    L = np.asarray(matriz_linkage, dtype=np.float64)
    N = L.shape[0] + 1
    miembros = {i: [i] for i in range(N)}
    usa_ids_estandar = np.max(L[:, :2]) >= N

    clusters_activos = set(range(N))

    for paso in range(N - 1):
        # Si cortamos por número de clusters y ya alcanzamos K, detenemos las fusiones
        if n_clusters is not None and len(clusters_activos) <= n_clusters:
            break

        c1 = int(L[paso, 0])
        c2 = int(L[paso, 1])
        distancia_fusion = L[paso, 2]

        # Si cortamos por altura y la fusión supera altura_h, detenemos las fusiones
        if altura_h is not None and distancia_fusion > altura_h:
            break

        fusionados = miembros[c1] + miembros[c2]
        clusters_activos.discard(c1)
        clusters_activos.discard(c2)

        nuevo_id = (N + paso) if usa_ids_estandar else c1
        miembros[nuevo_id] = fusionados
        clusters_activos.add(nuevo_id)

    # Asignar etiquetas enteras limpias 0, 1, ..., K-1 a cada instancia original
    etiquetas = np.zeros(N, dtype=int)
    for nuevo_label, cid in enumerate(sorted(clusters_activos)):
        for idx_punto in miembros[cid]:
            etiquetas[idx_punto] = nuevo_label

    return etiquetas


def optimizar_hiperparametros_internos(matriz_distancias, enlaces=("single", "complete", "average"), rango_k=range(2, 11)):
    """
    Banco de pruebas de validación interna (Scratch): evalúa los distintos criterios
    de enlace de Lance-Williams mediante CPCC y busca el número óptimo de clusters K
    maximizando el Silhouette Score.

    Pre-condiciones:
        - matriz_distancias: numpy.ndarray simétrica (N, N).
        - enlaces: iterable con los tipos de enlace a evaluar.
        - rango_k: iterable de enteros K (2 <= K < N).

    Post-condiciones:
        - Devuelve un diccionario con el resumen experimental y la configuración óptima.
    """
    from src.clustering.hierarchical import MotorJerarquico

    resultados = {}
    mejor_config = {
        "enlace": None,
        "k_optimo": None,
        "silhouette": -1.0,
        "cpcc": -1.0,
        "linkage_matrix": None,
        "etiquetas": None,
    }

    for enlace in enlaces:
        motor = MotorJerarquico(tipo_enlace=enlace)
        Z = motor.fit(matriz_distancias)
        cpcc = calcular_cpcc_scratch(matriz_distancias, Z)

        sil_por_k = {}
        for k in rango_k:
            etiquetas_k = cortar_dendrograma_scratch(Z, n_clusters=k)
            if len(np.unique(etiquetas_k)) < 2:
                continue
            det = calcular_silhouette_scratch(matriz_distancias, etiquetas_k, devolver_detalle=True)
            sil_por_k[k] = {
                "silhouette": det["Silhouette Score"],
                "cohesion_media": det["Cohesion Media a(i)"],
                "separacion_media": det["Separacion Media b(i)"],
            }

            if det["Silhouette Score"] > mejor_config["silhouette"]:
                mejor_config.update({
                    "enlace": enlace,
                    "k_optimo": k,
                    "silhouette": det["Silhouette Score"],
                    "cpcc": cpcc,
                    "linkage_matrix": Z,
                    "etiquetas": etiquetas_k,
                })

        resultados[enlace] = {
            "cpcc": cpcc,
            "silhouette_por_k": sil_por_k,
        }

    return {"resultados_por_enlace": resultados, "mejor_configuracion": mejor_config}


if __name__ == "__main__":
    print("=== Toy-Example: Validación Interna Scratch (CPCC y Silhouette) ===")
    from src.clustering.hierarchical import MotorJerarquico

    # Datos sintéticos con 3 grupos bien diferenciados (15 puntos en total)
    np.random.seed(42)
    g1 = np.random.normal(loc=0.0, scale=0.2, size=(5, 4))
    g2 = np.random.normal(loc=5.0, scale=0.2, size=(5, 4))
    g3 = np.random.normal(loc=10.0, scale=0.2, size=(5, 4))
    X_toy = np.vstack([g1, g2, g3])

    # Matriz de distancias euclídeas simple para el toy-example
    diff = X_toy[:, None, :] - X_toy[None, :, :]
    D_toy = np.sqrt(np.sum(diff ** 2, axis=-1))

    for enlace in ["single", "complete", "average"]:
        motor = MotorJerarquico(tipo_enlace=enlace)
        Z_toy = motor.fit(D_toy)
        cpcc_val = calcular_cpcc_scratch(D_toy, Z_toy)
        labels_k3 = cortar_dendrograma_scratch(Z_toy, n_clusters=3)
        det_sil = calcular_silhouette_scratch(D_toy, labels_k3, devolver_detalle=True)
        print(
            f"Enlace: {enlace:8s} | CPCC: {cpcc_val:.4f} | "
            f"Silhouette (K=3): {det_sil['Silhouette Score']:.4f} "
            f"(a(i)={det_sil['Cohesion Media a(i)']:.4f}, b(i)={det_sil['Separacion Media b(i)']:.4f})"
        )