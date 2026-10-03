import math
import pickle
import numpy as np


class VectorizadorTexto:
    """
    Convierte textos en una matriz de números X donde:
    - Cada fila es una instancia x^(t) (un documento).
    - Cada columna es un atributo (una palabra del vocabulario).
    """

    def __init__(self, tipo="bow"):
        self.tipo = tipo          # 'bow' para conteo simple, 'tfidf' para TF-IDF
        self.vocabulario = {}     # palabra -> índice de columna
        self.idf = {}             # palabra -> peso IDF

    def fit(self, textos):
        """
        Aprende el vocabulario a partir de los textos de entrenamiento.
        """
        todas_las_palabras = set()
        doc_freq = {}  # en cuántos documentos aparece cada palabra

        for doc in textos:
            palabras_doc = set(doc.split())
            todas_las_palabras.update(palabras_doc)
            for p in palabras_doc:
                doc_freq[p] = doc_freq.get(p, 0) + 1

        # Creamos el vocabulario ordenado alfabéticamente
        palabras_ordenadas = sorted(list(todas_las_palabras))
        self.vocabulario = {p: i for i, p in enumerate(palabras_ordenadas)}

        # Si usamos TF-IDF, calculamos el peso IDF: log(N / DF)
        n_docs = len(textos)
        for p, df in doc_freq.items():
            self.idf[p] = math.log((1 + n_docs) / (1 + df)) + 1

        return self

    def transform(self, textos):
        """
        Convierte una lista de textos en una matriz NumPy X de tamaño (N, V).
        """
        n_docs = len(textos)
        n_vocab = len(self.vocabulario)

        # Inicializamos la matriz X con ceros
        X = np.zeros((n_docs, n_vocab), dtype=np.float32)

        for i, doc in enumerate(textos):
            palabras = doc.split()
            for p in palabras:
                if p in self.vocabulario:
                    j = self.vocabulario[p]
                    if self.tipo == "bow":
                        X[i, j] += 1.0
                    elif self.tipo == "tfidf":
                        X[i, j] += 1.0 * self.idf.get(p, 1.0)

        # Si es TF-IDF, normalizamos los vectores para comparar distancias fácilmente
        if self.tipo == "tfidf":
            normas = np.linalg.norm(X, axis=1, keepdims=True)
            normas[normas == 0] = 1.0
            X = X / normas

        return X

    def fit_transform(self, textos):
        return self.fit(textos).transform(textos)


# --- Funciones sencillas de Persistencia (Guardar / Cargar) ---

def guardar_vectorizador(vectorizador, ruta_archivo):
    """Guarda el vocabulario y los pesos con pickle para usar en inferencia."""
    with open(ruta_archivo, "wb") as f:
        pickle.dump(vectorizador, f)
    print(f"Vectorizador guardado en: {ruta_archivo}")


def cargar_vectorizador(ruta_archivo):
    """Carga un vectorizador previamente guardado."""
    with open(ruta_archivo, "rb") as f:
        vectorizador = pickle.load(f)
    print(f"Vectorizador cargado con éxito. Talla vocabulario: {len(vectorizador.vocabulario)}")
    return vectorizador
