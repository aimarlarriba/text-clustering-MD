import os
import numpy as np
import matplotlib.pyplot as plt


# --- Embeddings con SBERT ---

def generar_embeddings(textos, nombre_modelo="all-MiniLM-L6-v2", normalizar=True):
    """
    Genera representaciones densas contextuales en R^(N x 384) usando Sentence-BERT.

    Pre-condiciones:
        - textos: lista o array con textos (strings).
        - nombre_modelo: identificador del modelo de sentence-transformers.
        - normalizar: si es True, normaliza los vectores a norma unitaria (L2 = 1.0).

    Post-condiciones:
        - Devuelve una matriz de numpy de dimensiones (N, 384) en float32.
    """
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(nombre_modelo)
    textos_limpios = [str(t) if t is not None else "" for t in textos]
    vectores = modelo.encode(
        textos_limpios,
        normalize_embeddings=normalizar,
        show_progress_bar=True
    )
    return np.asarray(vectores, dtype=np.float32)


def guardar_embeddings(X: np.ndarray, ruta_archivo: str) -> None:
    """
    Guarda la matriz de embeddings en un fichero .npy.

    Pre-condiciones:
        - X: matriz numpy de dimensión (N, D).
        - ruta_archivo: ruta del fichero de salida.

    Post-condiciones:
        - Archivo guardado en disco (crea carpetas si no existen).
    """
    directorio = os.path.dirname(os.path.abspath(ruta_archivo))
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    np.save(ruta_archivo, X)
    print(f"Embeddings guardados en: {ruta_archivo} (shape: {X.shape})")


def cargar_embeddings(ruta_archivo: str) -> np.ndarray:
    """
    Carga una matriz de embeddings previamente guardada en formato .npy.

    Pre-condiciones:
        - ruta_archivo: ruta a un fichero .npy válido.

    Post-condiciones:
        - Devuelve la matriz numpy cargada.
    """
    X = np.load(ruta_archivo)
    print(f"Embeddings cargados desde: {ruta_archivo} (shape: {X.shape})")
    return X


# --- Distancias individuales (Scratch) ---

def distancia_coseno_vector_scratch(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calcula la distancia de coseno entre dos vectores u y v desde cero.
    d_cos(u, v) = 1 - (u · v) / (||u||_2 * ||v||_2)

    Pre-condiciones:
        - u, v: vectores 1D de numpy de la misma dimensión D.

    Post-condiciones:
        - Devuelve un float en [0.0, 2.0].
    """
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0 if norm_u == norm_v else 1.0

    similitud = np.sum(u * v) / (norm_u * norm_v)
    similitud = np.clip(similitud, -1.0, 1.0)
    return float(np.clip(1.0 - similitud, 0.0, 2.0))


def distancia_euclidea_vector_scratch(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calcula la distancia euclídea entre dos vectores u y v desde cero.
    d_E(u, v) = sqrt(sum((u_i - v_i)^2))

    Pre-condiciones:
        - u, v: vectores 1D de numpy de la misma dimensión D.

    Post-condiciones:
        - Devuelve un float >= 0.0.
    """
    return float(np.sqrt(np.sum((u - v) ** 2)))


def distancia_minkowski_vector_scratch(u: np.ndarray, v: np.ndarray, p: float = 3.0) -> float:
    """
    Calcula la distancia de Minkowski de orden p entre dos vectores desde cero.
    d_p(u, v) = (sum(|u_i - v_i|^p))^(1/p)

    Pre-condiciones:
        - u, v: vectores 1D de numpy de la misma dimensión D.
        - p: float >= 1.0 (p=1 es Manhattan, p=2 es Euclídea).

    Post-condiciones:
        - Devuelve un float >= 0.0.
    """
    if p < 1.0:
        raise ValueError("El parámetro p de Minkowski debe ser mayor o igual a 1.")
    return float(np.sum(np.abs(u - v) ** p) ** (1.0 / p))


# --- Matrices de distancias N x N (Scratch vectorizado en NumPy) ---

def calcular_matriz_distancias_scratch(X: np.ndarray, metrica: str = "cosine", p: float = 2.0) -> np.ndarray:
    """
    Calcula la matriz cuadrada de distancias (N x N) por pares desde cero
    aprovechando operaciones matriciales de NumPy sin usar scipy ni sklearn.

    Pre-condiciones:
        - X: matriz bidimensional numpy de tamaño (N, D).
        - metrica: 'cosine', 'euclidean' o 'minkowski'.
        - p: valor de p si se usa Minkowski (por defecto 2.0).

    Post-condiciones:
        - Devuelve una matriz cuadrada simétrica (N, N) con ceros en la diagonal.
    """
    if X.ndim != 2:
        raise ValueError(f"X debe ser una matriz 2D. Recibida: {X.shape}")

    N, D = X.shape
    metrica_clean = metrica.lower().strip()

    if metrica_clean in ["cosine", "coseno"]:
        # Normalizamos por filas para calcular similitud mediante dot product: S = X_norm @ X_norm.T
        normas = np.linalg.norm(X, axis=1, keepdims=True)
        normas = np.where(normas == 0.0, 1.0, normas)
        X_norm = X / normas

        similitud = np.dot(X_norm, X_norm.T)
        similitud = np.clip(similitud, -1.0, 1.0)
        D_mat = 1.0 - similitud

    elif metrica_clean in ["euclidean", "euclidea"]:
        # Expandimos: ||u - v||^2 = ||u||^2 + ||v||^2 - 2(u @ v)
        # Esto permite calcular la matriz completa con dot product rápido de BLAS
        cuadrados = np.sum(X ** 2, axis=1, keepdims=True)
        D_cuad = cuadrados + cuadrados.T - 2.0 * np.dot(X, X.T)
        np.maximum(D_cuad, 0.0, out=D_cuad)  # evitar negativos por imprecision flotante
        D_mat = np.sqrt(D_cuad)

    elif metrica_clean == "minkowski":
        if p < 1.0:
            raise ValueError("El parámetro p para Minkowski debe ser >= 1.0.")
        if np.isclose(p, 2.0):
            return calcular_matriz_distancias_scratch(X, metrica="euclidean")

        # Para p != 2 usamos bloques de filas para no saturar memoria RAM
        D_mat = np.zeros((N, N), dtype=np.float32)
        bloque = 100
        for i in range(0, N, bloque):
            fin = min(i + bloque, N)
            diff = X[i:fin, np.newaxis, :] - X[np.newaxis, :, :]
            D_mat[i:fin, :] = np.sum(np.abs(diff) ** p, axis=-1) ** (1.0 / p)

    else:
        raise ValueError(f"Métrica '{metrica}' no soportada (usar 'cosine', 'euclidean' o 'minkowski').")

    # Aseguramos simetría exacta y diagonal nula
    D_mat = (D_mat + D_mat.T) / 2.0
    np.fill_diagonal(D_mat, 0.0)
    np.maximum(D_mat, 0.0, out=D_mat)

    return D_mat.astype(np.float32)


def guardar_matriz_distancias(D: np.ndarray, ruta_archivo: str) -> None:
    """
    Guarda la matriz cuadrada de distancias en un archivo .npy.
    """
    directorio = os.path.dirname(os.path.abspath(ruta_archivo))
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    np.save(ruta_archivo, D)
    print(f"Matriz de distancias guardada en: {ruta_archivo} (shape: {D.shape})")


def cargar_matriz_distancias(ruta_archivo: str) -> np.ndarray:
    """
    Carga una matriz cuadrada de distancias desde un archivo .npy.
    """
    D = np.load(ruta_archivo)
    print(f"Matriz de distancias cargada desde: {ruta_archivo} (shape: {D.shape})")
    return D


# --- Comparativa de espacios métricos ---

def comparar_espacios_metricos(
    X: np.ndarray,
    D_coseno: np.ndarray,
    D_euclidea: np.ndarray,
    ruta_grafico: str = None
) -> dict:
    """
    Calcula estadísticos descriptivos, métricas formales de contraste dimensional
    (Aggarwal et al., 2001) y la verificación experimental de la relación matemática
    d_E = sqrt(2 * d_cos) derivada de la normalización L2 unitaria.

    Pre-condiciones:
        - X: matriz (N, D) de embeddings.
        - D_coseno, D_euclidea: matrices simétricas (N, N) con diagonal 0.
        - ruta_grafico: opcional, ruta para guardar el gráfico comparativo.

    Post-condiciones:
        - Devuelve un diccionario con estadísticas, contraste relativo y error teórico.
    """
    N = D_coseno.shape[0]
    filas, cols = np.triu_indices(N, k=1)

    cos_pares = D_coseno[filas, cols]
    euc_pares = D_euclidea[filas, cols]

    cos_min = float(np.min(cos_pares))
    cos_max = float(np.max(cos_pares))
    euc_min = float(np.min(euc_pares))
    euc_max = float(np.max(euc_pares))

    # Contraste relativo (Relative Contrast - Aggarwal et al., 2001)
    # C_R = (dist_max - dist_min) / dist_min
    contraste_cos = (cos_max - cos_min) / (cos_min if cos_min > 0 else 1e-6)
    contraste_euc = (euc_max - euc_min) / (euc_min if euc_min > 0 else 1e-6)

    # Verificación empírica de la identidad formal: d_E = sqrt(2 * d_cos) si ||u|| = 1
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2(u · v) = 2 - 2*cos_sim = 2 * d_cos
    euc_teorica = np.sqrt(2.0 * np.clip(cos_pares, 0.0, 2.0))
    error_teorico_max = float(np.max(np.abs(euc_pares - euc_teorica)))
    error_teorico_medio = float(np.mean(np.abs(euc_pares - euc_teorica)))

    stats = {
        "n_muestras": N,
        "n_pares": len(cos_pares),
        "cos_media": float(np.mean(cos_pares)),
        "cos_std": float(np.std(cos_pares)),
        "cos_min": cos_min,
        "cos_max": cos_max,
        "cos_contraste_relativo": float(contraste_cos),
        "euc_media": float(np.mean(euc_pares)),
        "euc_std": float(np.std(euc_pares)),
        "euc_min": euc_min,
        "euc_max": euc_max,
        "euc_contraste_relativo": float(contraste_euc),
        "corr_pearson": float(np.corrcoef(cos_pares, euc_pares)[0, 1]),
        "error_formal_max_euc_sqrt2cos": error_teorico_max,
        "error_formal_medio_euc_sqrt2cos": error_teorico_medio,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Distribución Coseno
    axes[0].hist(cos_pares, bins=35, color="#2b5c8f", edgecolor="white", alpha=0.85)
    axes[0].set_title(f"Distancia Coseno (C_R={contraste_cos:.2f})")
    axes[0].set_xlabel("d_cos(u, v)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # 2. Distribución Euclídea
    axes[1].hist(euc_pares, bins=35, color="#d95f02", edgecolor="white", alpha=0.85)
    axes[1].set_title(f"Distancia Euclídea (C_R={contraste_euc:.2f})")
    axes[1].set_xlabel("d_E(u, v)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # 3. Dispersión Coseno vs Euclídea
    n_puntos = min(1500, len(cos_pares))
    idx = np.random.choice(len(cos_pares), size=n_puntos, replace=False)
    axes[2].scatter(cos_pares[idx], euc_pares[idx], alpha=0.35, color="#2ca02c", s=12)
    axes[2].set_title(f"Relación Coseno vs Euclídea (r = {stats['corr_pearson']:.4f})")
    axes[2].set_xlabel("Distancia Coseno")
    axes[2].set_ylabel("Distancia Euclídea")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if ruta_grafico:
        directorio = os.path.dirname(os.path.abspath(ruta_grafico))
        if directorio:
            os.makedirs(directorio, exist_ok=True)
        plt.savefig(ruta_grafico, dpi=200)
        print(f"Gráfico guardado en: {ruta_grafico}")

    plt.close(fig)
    return stats


def perfilar_coste_computacional(X: np.ndarray, lista_n=(100, 200, 300, 400, 500)) -> list:
    """
    Aporta evidencias cuantitativas del coste computacional (tiempo en segundos
    y memoria en MB) al calcular las matrices de distancia NxN para demostrar
    empíricamente la complejidad O(N^2 * D).

    Pre-condiciones:
        - X: matriz (N, D) con N >= max(lista_n).
        - lista_n: iterable de tamaños de muestra crecientes a evaluar.

    Post-condiciones:
        - Devuelve una lista de diccionarios con el profiling de tiempo y memoria.
    """
    import time
    resultados = []
    N_total, D = X.shape

    for n in lista_n:
        if n > N_total:
            continue
        sub_X = X[:n]

        # Tiempo Coseno
        t0 = time.perf_counter()
        _ = calcular_matriz_distancias_scratch(sub_X, metrica="cosine")
        t_cos = time.perf_counter() - t0

        # Tiempo Euclídea
        t0 = time.perf_counter()
        _ = calcular_matriz_distancias_scratch(sub_X, metrica="euclidean")
        t_euc = time.perf_counter() - t0

        # Memoria teórica de la matriz (float32 = 4 bytes)
        memoria_mb = (n * n * 4) / (1024 * 1024)

        resultados.append({
            "N": n,
            "Dimension_D": D,
            "Tiempo_Coseno_seg": round(t_cos, 5),
            "Tiempo_Euclidea_seg": round(t_euc, 5),
            "Memoria_Matriz_MB": round(memoria_mb, 4)
        })

    return resultados


# --- BLOQUE DE PRUEBA ---
if __name__ == "__main__":
    print("Probando módulo de embeddings y distancias scratch...")

    # Generamos datos dummy para comprobar
    X_test = np.random.randn(20, 384).astype(np.float32)

    D_cos = calcular_matriz_distancias_scratch(X_test, metrica="cosine")
    D_euc = calcular_matriz_distancias_scratch(X_test, metrica="euclidean")
    print(f"Matrices calculadas correctamente: Coseno {D_cos.shape}, Euclídea {D_euc.shape}")

    # Comprobamos rápidamente frente a scipy por seguridad
    try:
        from scipy.spatial.distance import cdist
        diff_cos = np.max(np.abs(D_cos - cdist(X_test, X_test, metric="cosine")))
        diff_euc = np.max(np.abs(D_euc - cdist(X_test, X_test, metric="euclidean")))
        if diff_cos < 1e-4 and diff_euc < 1e-4:
            print("Sanity check con SciPy: OK (cálculo scratch idéntico)")
        else:
            print("Aviso: pequeñas discrepancias con SciPy:", diff_cos, diff_euc)
    except ImportError:
        pass
