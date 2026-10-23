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
    Motor Jerárquico Aglomerativo (SAHN) implementado desde cero.
    """

    def __init__(self, tipo_enlace="single"):
        """
        Inicializa el motor de clustering.
        
        :param tipo_enlace: Criterio de enlace a usar: 'single' (min), 'complete' (max) o 'average' (promedio).
        """
        if tipo_enlace not in ["single", "complete", "average"]:
            raise ValueError("El tipo_enlace debe ser 'single', 'complete' o 'average'")
        self.tipo_enlace = tipo_enlace
        self.linkage_matrix = None

    def fit(self, matriz_distancias):
        """
        Ejecuta el algoritmo aglomerativo iterativo sobre una matriz de distancias precalculada.
        
        :param matriz_distancias: Matriz cuadrada de distancias (N x N).
        :return: linkage_matrix de tamaño (N-1, 4) en formato estándar compatible con SciPy.
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

            # 4. Actualizar la matriz de distancias usando Lance-Williams
            self._actualizar_distancias(D, cluster1, cluster2, tamanos_matriz)
            tamanos_matriz[cluster1] = n_muestras

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
        """
        actualizar_matriz_lance_williams(D, cluster1, cluster2, self.tipo_enlace, tamanos_matriz)


        
