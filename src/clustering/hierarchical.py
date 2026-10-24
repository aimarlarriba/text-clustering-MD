import numpy as np
import time
try:
    from src.clustering.lance_williams import actualizar_matriz_lance_williams
except (ImportError, ModuleNotFoundError):
    try:
        from .lance_williams import actualizar_matriz_lance_williams
    except (ImportError, ModuleNotFoundError):
        from clustering.lance_williams import actualizar_matriz_lance_williams


class MotorJerarquico:
    """
    Motor Jerárquico Aglomerativo (SAHN: Sequential Agglomerative Hierarchical Non-overlapping)
    implementado desde cero (Scratch) mediante la fórmula recursiva de Lance-Williams (1967).
    """

    def __init__(self, tipo_enlace="single"):
        """
        Inicializa el motor de clustering jerárquico.

        Pre-condiciones:
            - tipo_enlace: str en {'single', 'complete', 'average'}.

        Post-condiciones:
            - Instancia configurada con el enlace seleccionado y linkage_matrix inicializada en None.
        """
        if tipo_enlace not in ["single", "complete", "average"]:
            raise ValueError("El tipo_enlace debe ser 'single', 'complete' o 'average'")
        self.tipo_enlace = tipo_enlace
        self.linkage_matrix = None

    def fit(self, matriz_distancias):
        """
        Ejecuta el algoritmo aglomerativo iterativo sobre una matriz de distancias precalculada.

        Pre-condiciones:
            - matriz_distancias: numpy.ndarray simétrica cuadrada (N x N) con distancias d(i, j) >= 0,
              diagonal nula d(i, i) = 0 y N >= 2 instancias.

        Post-condiciones:
            - Devuelve y almacena self.linkage_matrix: numpy.ndarray de dimensiones (N - 1, 4)
              en formato canónico compatible con SciPy:
              [id_cluster_1, id_cluster_2, distancia_fusion, tamano_cluster_resultante].
            - Invariante de iteración: en el paso p (0 <= p < N-1), existen exactamente N - p clusters
              activos disjuntos particionando el conjunto total de instancias.
        """
        start_time = time.time()
        
        # 1. Copia de trabajo para no destruir los datos originales
        D = np.array(matriz_distancias, dtype=float, copy=True)
        N = D.shape[0]
        
        # 2. Ignoramos la diagonal para no fusionar una muestra consigo misma
        np.fill_diagonal(D, np.inf)
        
        # 3. Matriz de enlace de salida (N-1 pasos, 4 columnas)
        self.linkage_matrix = np.zeros((N - 1, 4), dtype=float)
        
        # 4. Diccionarios de seguimiento:
        # - cluster_id_activo: mapea la fila i de la matriz con su ID global de cluster actual
        # - tamano_clusters: guarda cuántas muestras contiene cada ID de cluster
        cluster_id_activo = {i: i for i in range(N)}
        tamano_clusters = {i: 1 for i in range(N)}

        # Inicializamos los tamaños de cada fila de la matriz para Lance-Williams
        tamanos_matriz = {i: 1 for i in range(N)}

        for paso in range(N - 1):
            # 1. Encontrar los dos clusters más cercanos (mínima distancia en D)
            idx_min = np.argmin(D)
            cluster1, cluster2 = np.unravel_index(idx_min, D.shape)
            distancia_minima = float(D[cluster1, cluster2])

            if cluster1 > cluster2:
                cluster1, cluster2 = cluster2, cluster1

            # 2. Identificar qué clusters reales están viviendo en esas posiciones
            id1 = cluster_id_activo[cluster1]
            id2 = cluster_id_activo[cluster2]

            # 3. Crear el nuevo cluster con ID global (N + paso) y registrar su tamaño
            nuevo_id_cluster = N + paso
            n_muestras = tamano_clusters[id1] + tamano_clusters[id2]
            tamano_clusters[nuevo_id_cluster] = n_muestras

            # 4. Actualizar la matriz de distancias y el tamaño del cluster fusionado usando Lance-Williams
            self._actualizar_distancias(D, cluster1, cluster2, tamanos_matriz)

            # 5. Guardar la fusión en linkage_matrix
            self.linkage_matrix[paso] = [id1, id2, distancia_minima, n_muestras]

            # 6. Actualizar el traductor: la fila cluster1 ahora representa a nuevo_id_cluster
            cluster_id_activo[cluster1] = nuevo_id_cluster


        
        end_time = time.time()
        print(f"Ejecución completada en {end_time - start_time:.4f} segundos.")
        
        return self.linkage_matrix

    def _actualizar_distancias(self, D, cluster1, cluster2, tamanos_matriz):
        """
        Actualiza la matriz de distancias tras fusionar cluster1 y cluster2
        aplicando la fórmula recursiva de Lance-Williams.

        Pre-condiciones:
            - D: numpy.ndarray cuadrada (N x N) modificable in-place.
            - cluster1, cluster2: índices enteros válidos activos en la matriz.
            - tamanos_matriz: diccionario con los tamaños actuales de los clusters.

        Post-condiciones:
            - Actualiza D[cluster1, :] y D[:, cluster1] con d(C1 U C2, Ck) y anula cluster2 con inf.
        """
        actualizar_matriz_lance_williams(D, cluster1, cluster2, self.tipo_enlace, tamanos_matriz)


        
