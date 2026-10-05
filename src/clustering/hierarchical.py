import numpy as np
import time

class MotorJerarquico:
    """
    Motor Jerárquico Aglomerativo (SAHN) implementado desde cero.
    Desarrollado por: Urko
    """

    def __init__(self, tipo_enlace="single"):
        """
        Inicializa el motor de clustering.
        
        :param tipo_enlace: Criterio de enlace a usar: 'single' (min) o 'complete' (max).
        """
        if tipo_enlace not in ["single", "complete"]:
            raise ValueError("El tipo_enlace debe ser 'single' o 'complete'")
        self.tipo_enlace = tipo_enlace
        self.linkage_matrix = None

    def fit(self, matriz_distancias):
        """
        Ejecuta el algoritmo aglomerativo iterativo sobre una matriz de distancias precalculada.
        
        :param matriz_distancias: Matriz cuadrada de distancias (N x N)
        :return: linkage_matrix de tamaño (N-1, 4)
        """
        # TODO: Implementar el bucle iterativo:
        # 1. Inicializar N clusters unitarios.
        # 2. Encontrar los dos clusters i, j más cercanos (minima distancia en la matriz, ignorando la diagonal 0).
        # 3. Fusionar i y j creando un nuevo cluster.
        # 4. Actualizar la matriz de distancias según el criterio (Single-link o Complete-link).
        # 5. Guardar la fusión en self.linkage_matrix: [idx1, idx2, dist, n_muestras].
        # 6. Repetir hasta que quede 1 solo cluster.

        # Inicio del profiling de tiempo
        start_time = time.time()
        
        N = matriz_distancias.shape[0]
        self.linkage_matrix = np.zeros((N - 1, 4))
        
        # Llevaremos la cuenta del tamaño de cada cluster para la columna 4.
        # Al principio, los clusters 0 a N-1 tienen tamaño 1.
        tamano_clusters = {i: 1 for i in range(N)}

        for paso in range(N-1):
            distancia_minima = np.inf
            cluster1 = -1
            cluster2 = -1

            # Encontrar los dos clusters más cercanos
            for i in range(N):
                for j in range(i+1, N):
                    if matriz_distancias[i, j] < distancia_minima:
                        distancia_minima = matriz_distancias[i, j]
                        cluster1 = i
                        cluster2 = j

            # El ID del nuevo cluster según el estándar es N + paso
            nuevo_id_cluster = N + paso
            
            # El número de muestras es la suma de las muestras de los dos clusters originales
            n_muestras = tamano_clusters[cluster1] + tamano_clusters[cluster2]
            tamano_clusters[nuevo_id_cluster] = n_muestras

            #Actualizar la matriz de distancias
            self._actualizar_distancias(matriz_distancias, cluster1, cluster2)

            # Guardar la fusión en linkage_matrix
            self.linkage_matrix[paso] = [cluster1, cluster2, distancia_minima, n_muestras]

        
        end_time = time.time()
        print(f"Ejecución completada en {end_time - start_time:.4f} segundos.")
        
        return self.linkage_matrix

    def _actualizar_distancias(self, matriz_distancias, cluster1, cluster2):
        """
        Actualiza la matriz de distancias tras fusionar cluster1 y cluster2.
        
        Aquí implementarás las ecuaciones de Single-link (min) y Complete-link (max).
        """
        pass
