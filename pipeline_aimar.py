"""
Pipeline de preparación de datos y matrices de distancias (Aimar).
Genera embeddings contextuales con SBERT y matrices de distancia (Scratch)
para el resto de compañeros del equipo (Urko, Adrián, María).
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd

from src.embeddings import (
    generar_embeddings,
    guardar_embeddings,
    calcular_matriz_distancias_scratch,
    guardar_matriz_distancias,
    comparar_espacios_metricos,
)


def crear_dataset_sintetico_cssrs(n_muestras=500):
    """
    Crea un benchmark reproducible con 500 posts basados en las 5 clases
    de riesgo de C-SSRS (Gaur et al., 2019):
    0: Indicator, 1: Ideation, 2: Behavior, 3: Attempt, 4: Supportive
    """
    np.random.seed(42)

    plantillas = {
        "Indicator": [
            "I feel so empty lately, nothing brings me joy anymore.",
            "Feeling overwhelmed with constant sadness and exhaustion.",
            "Life feels pointless, like I am just drifting through the days without purpose.",
            "I cannot find any motivation to get out of bed in the morning.",
            "Everything seems gray and lifeless, I just want the pain to stop feeling so heavy."
        ],
        "Ideation": [
            "I keep thinking that everyone would be much better off without me around.",
            "I have thoughts of ending it all, though I don't know if I have the courage.",
            "What if I just disappeared forever? Would anyone even notice or care?",
            "I keep having intrusive thoughts about not waking up tomorrow.",
            "Thinking constantly about suicide as my only exit from this ongoing misery."
        ],
        "Behavior": [
            "I started giving away my favorite books and personal belongings to friends.",
            "I wrote a farewell letter to my family last night just in case.",
            "I have been researching methods and looking at tall bridges near my town.",
            "I went out and bought a large quantity of pills and hid them in my room.",
            "Saying my final goodbyes to people who meant something to me over the years."
        ],
        "Attempt": [
            "I survived my overdose yesterday and woke up in the emergency hospital.",
            "I tried hanging myself last week but the belt snapped at the last second.",
            "I woke up in the ICU after cutting too deep, feeling completely broken.",
            "I attempted suicide two days ago and now my parents won't leave me alone.",
            "Took a handful of pills yesterday, threw up all night and nobody knows."
        ],
        "Supportive": [
            "You are not alone in this fight, please reach out to the crisis helpline.",
            "I have been where you are and things truly do get better with therapy.",
            "Sending you a warm hug, please hold on and talk to someone who cares.",
            "Your life has immense value, please don't give up right now.",
            "We are here to listen whenever you need to vent without any judgment."
        ]
    }

    clases = list(plantillas.keys())
    textos = []
    etiquetas = []

    muestras_por_clase = n_muestras // len(clases)

    for idx_clase, clase in enumerate(clases):
        frases = plantillas[clase]
        for _ in range(muestras_por_clase):
            base = np.random.choice(frases)
            variacion = f" [post_{len(textos) + 1}]"
            textos.append(base + variacion)
            etiquetas.append(idx_clase)

    while len(textos) < n_muestras:
        c = np.random.choice(len(clases))
        textos.append(np.random.choice(plantillas[clases[c]]))
        etiquetas.append(c)

    return textos, np.array(etiquetas, dtype=int)


def limpiar_texto_post(post_raw):
    """
    En el dataset de Zenodo, la columna Post contiene una lista en formato string: "['post1', 'post2']".
    Esta función extrae y concatena todos los textos del usuario en un único string limpio.
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


def cargar_datos(ruta_csv=None, n_muestras=500):
    """
    Carga el CSV de Zenodo (500_Reddit_users_posts_labels.csv) si existe o genera
    el benchmark reproducible de prueba.
    """
    mapa_clases = {
        "indicator": 0,
        "ideation": 1,
        "behavior": 2,
        "attempt": 3,
        "supportive": 4
    }

    if ruta_csv and os.path.exists(ruta_csv):
        print(f"Cargando dataset desde {ruta_csv}...")
        df = pd.read_csv(ruta_csv)

        # Buscar columna de texto (en Zenodo se llama 'Post')
        col_texto = None
        for c in df.columns:
            if c.lower() in ["post", "text", "body", "cleaned_text"]:
                col_texto = c
                break
        if not col_texto:
            col_texto = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        # Buscar columna de etiqueta (en Zenodo se llama 'Label')
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
            print(f"Posts cargados: {len(textos)} | Distribución de clases: {distribucion}")
        else:
            etiquetas = np.zeros(len(textos), dtype=int)

        return textos, etiquetas

    print(f"No se indicó CSV o no existe. Usando benchmark de {n_muestras} posts de C-SSRS...")
    return crear_dataset_sintetico_cssrs(n_muestras)


def main():
    parser = argparse.ArgumentParser(description="Pipeline de embeddings y distancias (Aimar)")
    parser.add_argument("--csv", type=str, default=None, help="Ruta al CSV con los posts (opcional)")
    parser.add_argument("--n_muestras", type=int, default=500, help="Número de posts a procesar")
    parser.add_argument("--modelo", type=str, default="all-MiniLM-L6-v2", help="Modelo de SBERT")
    parser.add_argument("--salida", type=str, default="data_processed", help="Carpeta de salida")
    args = parser.parse_args()

    os.makedirs(args.salida, exist_ok=True)

    print("=== Pipeline de Embeddings y Matrices de Distancia (Aimar) ===")

    # 1. Cargar textos
    textos, etiquetas_reales = cargar_datos(args.csv, args.n_muestras)
    N = len(textos)
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
    print(f"Media Coseno: {stats['cos_media']:.4f} (std: {stats['cos_std']:.4f})")
    print(f"Media Euclídea: {stats['euc_media']:.4f} (std: {stats['euc_std']:.4f})")
    print(f"Correlación Pearson: {stats['corr_pearson']:.4f}")

    # 5. Guardar archivos para Urko, Adrián y María
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
