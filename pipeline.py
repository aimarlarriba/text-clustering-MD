"""
Descripción:
    Pipeline integral que carga el corpus clínico de Reddit (Zenodo C-SSRS),
    genera representaciones densas contextuales mediante Sentence-BERT (all-MiniLM-L6-v2),
    calcula las matrices de distancias Coseno y Euclídea desde cero (Scratch en NumPy),
    evalúa formalmente el contraste métrico (Aggarwal et al., 2001) y perfila el coste computacional.
"""

import os
import sys
import time
import json
import argparse
from typing import Tuple, List, Any
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from src.config import (
        RAW_DATA_FILE,
        PROCESSED_DATA_DIR,
        DEFAULT_MODEL_NAME,
        DEFAULT_N_SAMPLES,
        CLASS_MAP,
    )
except (ImportError, ModuleNotFoundError):
    RAW_DATA_FILE = "data/500_Reddit_users_posts_labels.csv"
    PROCESSED_DATA_DIR = "data_processed"
    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_N_SAMPLES = 500
    CLASS_MAP = {
        "indicator": 0,
        "ideation": 1,
        "behavior": 2,
        "attempt": 3,
        "supportive": 4,
    }

from src.embeddings import (
    generar_embeddings,
    guardar_embeddings,
    calcular_matriz_distancias_scratch,
    guardar_matriz_distancias,
    comparar_espacios_metricos,
    perfilar_coste_computacional,
)



def limpiar_texto_post(post_raw: Any) -> str:
    """
    Extrae y unifica las publicaciones textuales de un usuario en un único string limpio.
    En el corpus de Zenodo, los posts múltiples se almacenan como una lista serializada en texto.

    Pre-condiciones:
        - post_raw: objeto convertible a string o representación literal de lista con los textos del post.

    Post-condiciones:
        - Devuelve un string no vacío con el contenido textual concatenado y normalizado.
    """
    import ast
    if isinstance(post_raw, str) and post_raw.strip().startswith("["):
        try:
            posts = ast.literal_eval(post_raw)
            if isinstance(posts, list):
                return " ".join([str(p).strip() for p in posts if p])
        except Exception:
            pass
    return str(post_raw).strip()


def cargar_datos(ruta_csv: str = "data/500_Reddit_users_posts_labels.csv", n_muestras: int = 500) -> Tuple[List[str], np.ndarray]:
    """
    Carga el dataset clínico real de Zenodo (500_Reddit_users_posts_labels.csv).

    Pre-condiciones:
        - ruta_csv: ruta al archivo CSV existente con los datos reales.
        - n_muestras: número máximo de usuarios a procesar (por defecto 500).

    Post-condiciones:
        - Devuelve una tupla (textos, etiquetas) donde textos es una lista de strings
          y etiquetas es un array NumPy con las clases de riesgo C-SSRS [0..4].
    """
    if not os.path.exists(ruta_csv):
        print(f"\n[AVISO] No se encontró el dataset en '{ruta_csv}'.")
        print("Asegúrate de colocar '500_Reddit_users_posts_labels.csv' dentro de la carpeta 'data/'.")
        return [], np.array([], dtype=int)

    mapa_clases = {
        "indicator": 0,
        "ideation": 1,
        "behavior": 2,
        "attempt": 3,
        "supportive": 4
    }

    print(f"Cargando dataset real desde {ruta_csv}...")
    df = pd.read_csv(ruta_csv)

    # Identificar columna de texto ('Post' en Zenodo)
    col_texto = None
    for c in df.columns:
        if c.lower() in ["post", "text", "body", "cleaned_text"]:
            col_texto = c
            break
    if not col_texto:
        col_texto = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Identificar columna de etiqueta ('Label' en Zenodo)
    col_label = None
    for c in df.columns:
        if c.lower() in ["label", "class", "risk_level", "risk"]:
            col_label = c
            break

    df_sub = df.head(n_muestras)
    textos = [limpiar_texto_post(p) for p in df_sub[col_texto]]

    if col_label:
        etiquetas_str = df_sub[col_label].astype(str).str.strip().str.lower()
        etiquetas = np.array([mapa_clases.get(lbl, 0) for lbl in etiquetas_str], dtype=int)
        distribucion = df_sub[col_label].value_counts().to_dict()

        duplicados = df_sub.duplicated(subset=[col_texto]).sum()
        missing = df_sub[col_texto].isna().sum()
        palabras_unicas = len(set(" ".join(textos).lower().split()))

        print(f"Posts cargados: {len(textos)} | Distribución de clases: {distribucion}")
        print(
            f"Duplicados detectados: {duplicados} | Missing values: {missing} | Talla vocabulario (aprox): {palabras_unicas}")
    else:
        etiquetas = np.zeros(len(textos), dtype=int)

    return textos, etiquetas


def main():
    parser = argparse.ArgumentParser(description="Pipeline de embeddings y matrices de proximidad")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/500_Reddit_users_posts_labels.csv",
        help="Ruta al CSV con los posts reales de Zenodo"
    )
    parser.add_argument("--n_muestras", type=int, default=500, help="Número de posts a procesar")
    parser.add_argument("--modelo", type=str, default="all-MiniLM-L6-v2", help="Modelo de SBERT")
    parser.add_argument("--salida", type=str, default="data_processed", help="Carpeta de salida")
    parser.add_argument("--texto", type=str, default=None, help="Introduce un texto en inglés para evaluar su riesgo.")
    parser.add_argument("--infer", action="store_true", help="Activa el modo de inferencia.")
    args = parser.parse_args()

    if args.texto or args.infer:
        texto_evaluar = args.texto if args.texto else "I feel so overwhelmed and I don't know what to do."
        print(f"\n[MODO INFERENCIA] Procesando nuevo texto: '{texto_evaluar}'")

        from src.evaluacion import inferir_y_mostrar_cercanos

        try:
            X_train = np.load(os.path.join(args.salida, "embeddings_sbert.npy"))
            # Corrección de fuga: se cargan los clústeres inferidos, no el ground truth clínico
            y_train = np.load(os.path.join(args.salida, "etiquetas_predichas.npy"))
            with open(os.path.join(args.salida, "textos_posts.json"), "r", encoding="utf-8") as f:
                textos_train = json.load(f)
        except FileNotFoundError:
            print("Error: Datos base o clústeres no encontrados. Ejecuta pipeline.py y evaluacion.py primero.")
            return

        vector_nuevo = generar_embeddings([texto_evaluar], nombre_modelo=args.modelo, normalizar=True)[0]
        inferir_y_mostrar_cercanos(vector_nuevo, X_train, y_train, textos_train, k=3)
        return

        vector_nuevo = generar_embeddings([texto_evaluar], nombre_modelo=args.modelo, normalizar=True)[0]
        inferir_y_mostrar_cercanos(vector_nuevo, X_train, y_train, textos_train, k=3)
        return

    os.makedirs(args.salida, exist_ok=True)

    print("=== Pipeline de Embeddings y Matrices de Proximidad ===")

    # 1. Cargar textos
    textos, etiquetas_reales = cargar_datos(args.csv, args.n_muestras)
    N = len(textos)
    if N == 0:
        print("[ERROR] No hay textos para procesar. Revisa la ruta del dataset. Abortando.")
        return
    print(f"Total de posts a procesar: {N}")

    # 2. Generar embeddings
    print(f"Generando embeddings con {args.modelo}...")
    t0 = time.time()
    X = generar_embeddings(textos, nombre_modelo=args.modelo, normalizar=True)
    t_emb = time.time() - t0
    print(f"Embeddings listos en {t_emb:.2f}s (forma: {X.shape})")

    # 3. Calcular matrices de proximidad scratch en NumPy
    print("Calculando matrices de distancia en scratch...")
    t0 = time.time()
    D_cos = calcular_matriz_distancias_scratch(X, metrica="cosine")
    t_cos = time.time() - t0

    t0 = time.time()
    D_euc = calcular_matriz_distancias_scratch(X, metrica="euclidean")
    t_euc = time.time() - t0

    print(f"Matriz Coseno calculada en {t_cos:.4f}s")
    print(f"Matriz Euclídea calculada en {t_euc:.4f}s")

    # 4. Comparativa de distancias para el informe
    print("Generando comparativa estadística de métricas...")
    ruta_grafico = os.path.join(args.salida, "comparativa_distancias.png")
    stats = comparar_espacios_metricos(X, D_cos, D_euc, ruta_grafico=ruta_grafico)
    print(f"Media Coseno: {stats['cos_media']:.4f} (std: {stats['cos_std']:.4f}) | Contraste: {stats['cos_contraste_relativo']:.2f}")
    print(f"Media Euclídea: {stats['euc_media']:.4f} (std: {stats['euc_std']:.4f}) | Contraste: {stats['euc_contraste_relativo']:.2f}")
    print(f"Correlación Pearson: {stats['corr_pearson']:.4f}")
    print(f"Verificación formal d_E = sqrt(2*d_cos) -> Error máx: {stats['error_formal_max_euc_sqrt2cos']:.2e}")

    # 5. Profiling cuantitativo de tiempo y memoria (O(N^2))
    print("Ejecutando profiling empírico de complejidad temporal y memoria...")
    profiling = perfilar_coste_computacional(X, lista_n=(100, 200, 300, 400, 500))

    # Guardar métricas cuantitativas para el informe
    informe_cuantitativo = {
        "estadisticas_espacios_metricos": stats,
        "profiling_complejidad_computacional": profiling
    }
    with open(os.path.join(args.salida, "metricas_espacios_y_profiling.json"), "w", encoding="utf-8") as f:
        json.dump(informe_cuantitativo, f, ensure_ascii=False, indent=2)

    # 6. Guardar archivos serializados para las fases de clustering y evaluación
    print("Guardando archivos serializados...")
    guardar_embeddings(X, os.path.join(args.salida, "embeddings_sbert.npy"))
    guardar_matriz_distancias(D_cos, os.path.join(args.salida, "matriz_distancias_cosine.npy"))
    guardar_matriz_distancias(D_euc, os.path.join(args.salida, "matriz_distancias_euclidean.npy"))

    np.save(os.path.join(args.salida, "etiquetas_reales.npy"), etiquetas_reales)
    with open(os.path.join(args.salida, "textos_posts.json"), "w", encoding="utf-8") as f:
        json.dump(textos, f, ensure_ascii=False, indent=2)

    print("\nListo. Todos los ficheros generados correctamente en:", args.salida)


if __name__ == "__main__":
    main()
