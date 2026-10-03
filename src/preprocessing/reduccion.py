import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def reducir_a_2d(X):
    """
    Reduce la matriz de datos X a 2 dimensiones principales mediante PCA
    para poder visualizarla en un plano (eje X y eje Y).
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    return X_2d, pca


def visualizar_puntos_2d(X_2d, titulos=None, nombre_grafico="Visualización 2D"):
    """
    Dibuja los textos proyectados en un plano 2D con Matplotlib.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], color="royalblue", s=70)

    # Si pasamos etiquetas o frases cortas, las anotamos junto a los puntos
    if titulos is not None:
        for i, texto in enumerate(titulos):
            resumen = texto[:25] + "..." if len(texto) > 25 else texto
            plt.annotate(resumen, (X_2d[i, 0] + 0.02, X_2d[i, 1] + 0.02), fontsize=8)

    plt.title(nombre_grafico)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
