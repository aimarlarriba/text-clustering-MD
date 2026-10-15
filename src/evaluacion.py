import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster


def calcular_metricas_externas(etiquetas_reales, etiquetas_predichas):
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


# --- BLOQUE DE PRUEBA ---
if __name__ == "__main__":
    print("Iniciando prueba del módulo...")
    # Simulamos 50 vectores de 384 dimensiones (como los que hará Aimar)
    vectores_mock = np.random.rand(50, 384)
    # Simulamos etiquetas aleatorias (5 clases como en vuestro README)
    reales_mock = np.random.randint(0, 5, size=50)
    predicciones_mock = np.random.randint(0, 5, size=50)

    resultados = calcular_metricas_externas(reales_mock, predicciones_mock)
    print("Cálculo de métricas:", resultados)
    print("Abriendo gráfico PCA...")
    visualizar_pca_clusters(vectores_mock, predicciones_mock)